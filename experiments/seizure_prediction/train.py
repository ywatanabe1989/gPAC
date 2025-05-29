import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import os
import logging
from typing import Dict, List
import numpy as np
from tqdm import tqdm

from utils.data_loader import create_dataloaders
from models.seizure_predictor import SeizurePredictor

class Trainer:
    def __init__(self, config_path: str):
        """
        Trainer class for seizure prediction model
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize model and move to GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SeizurePredictor(config_path).to(self.device)
        
        # Setup multi-GPU if available
        if torch.cuda.device_count() > 1 and len(self.config['gpu']['devices']) > 1:
            self.logger.info(f"Using {len(self.config['gpu']['devices'])} GPUs")
            self.model = nn.DataParallel(self.model, device_ids=self.config['gpu']['devices'])
            
        # Initialize optimizer and scheduler
        self.optimizer, self.scheduler = self.model.configure_optimizers()
        
        # Initialize criterion
        self.criterion = nn.BCELoss()
        
        # Setup automatic mixed precision if requested
        self.scaler = torch.cuda.amp.GradScaler() if self.config['gpu']['precision'] == 'mixed' else None
        
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        predictions = []
        targets = []
        
        with tqdm(dataloader, desc='Training') as pbar:
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                
                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        output = self.model(data)
                        loss = self.criterion(output, target)
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    loss.backward()
                    self.optimizer.step()
                
                total_loss += loss.item()
                predictions.extend(output.detach().cpu().numpy())
                targets.extend(target.cpu().numpy())
                
                pbar.set_postfix({'loss': loss.item()})
        
        metrics = {
            'loss': total_loss / len(dataloader),
            'auc': self._calculate_auc(predictions, targets)
        }
        
        return metrics
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for data, target in tqdm(dataloader, desc='Validation'):
                data, target = data.to(self.device), target.to(self.device)
                
                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        output = self.model(data)
                        loss = self.criterion(output, target)
                else:
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                total_loss += loss.item()
                predictions.extend(output.cpu().numpy())
                targets.extend(target.cpu().numpy())
        
        metrics = {
            'loss': total_loss / len(dataloader),
            'auc': self._calculate_auc(predictions, targets)
        }
        
        return metrics
    
    def _calculate_auc(self, predictions: List[float], targets: List[float]) -> float:
        """Calculate Area Under the ROC Curve"""
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(targets, predictions)
    
    def train(self, data_dir: str):
        """Main training loop"""
        # Create dataloaders
        dataloaders = create_dataloaders(data_dir, self.config_path)
        
        # Training loop
        best_val_auc = 0
        patience_counter = 0
        
        for epoch in range(self.config['training']['epochs']):
            self.logger.info(f"Epoch {epoch+1}/{self.config['training']['epochs']}")
            
            # Train
            train_metrics = self.train_epoch(dataloaders['train'])
            self.logger.info(f"Training metrics: {train_metrics}")
            
            # Validate
            val_metrics = self.validate(dataloaders['val'])
            self.logger.info(f"Validation metrics: {val_metrics}")
            
            # Update learning rate
            self.scheduler.step(val_metrics['loss'])
            
            # Early stopping
            if val_metrics['auc'] > best_val_auc:
                best_val_auc = val_metrics['auc']
                patience_counter = 0
                self._save_checkpoint(epoch, val_metrics)
            else:
                patience_counter += 1
                
            if patience_counter >= self.config['training']['patience']:
                self.logger.info("Early stopping triggered")
                break
    
    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }
        
        checkpoint_path = os.path.join(
            'experiments/seizure_prediction/results',
            f'checkpoint_epoch_{epoch}.pt'
        )
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    args = parser.parse_args()
    
    trainer = Trainer(args.config)
    trainer.train(args.data_dir)