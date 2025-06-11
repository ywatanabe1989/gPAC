#!/usr/bin/env python3
"""
SciTeX: Modern Scientific Paper Management System
A flexible, Python-based LaTeX manuscript builder for scientific publications.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml


class SciTeX:
    """Main class for scientific paper management."""
    
    def __init__(self, config_path: str = "scitex.yaml"):
        self.root_dir = Path.cwd()
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._setup_directories()
    
    def _load_config(self) -> dict:
        """Load configuration from YAML or JSON."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                if self.config_path.suffix == '.yaml':
                    return yaml.safe_load(f)
                else:
                    return json.load(f)
        else:
            # Create default config
            default_config = self._get_default_config()
            self._save_config(default_config)
            return default_config
    
    def _get_default_config(self) -> dict:
        """Get default configuration."""
        return {
            'project': {
                'name': 'gPAC',
                'title': 'gPAC: GPU-Accelerated Phase-Amplitude Coupling Analysis',
                'authors': [],
                'version': '1.0.0',
            },
            'directories': {
                'manuscript': 'manuscript',
                'revision': 'revision',
                'supplementary': 'supplementary',
                'figures': 'figures',
                'tables': 'tables',
                'build': 'build',
                'templates': 'templates',
            },
            'latex': {
                'engine': 'pdflatex',
                'bibtex': 'bibtex',
                'options': ['-shell-escape', '-interaction=nonstopmode'],
            },
            'output': {
                'formats': ['pdf', 'docx'],
                'versioning': True,
                'timestamp': True,
            },
            'features': {
                'word_count': True,
                'diff': True,
                'version_control': True,
                'auto_bibliography': True,
            }
        }
    
    def _save_config(self, config: dict):
        """Save configuration."""
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    def _setup_directories(self):
        """Setup required directories."""
        for dir_type, dir_name in self.config['directories'].items():
            dir_path = self.root_dir / dir_name
            dir_path.mkdir(exist_ok=True)
    
    def compile(self, target: str = 'manuscript', clean: bool = False, 
                view: bool = False, **kwargs):
        """Compile LaTeX document."""
        print(f"Compiling {target}...")
        
        target_dir = self.root_dir / self.config['directories'][target]
        main_file = target_dir / 'main.tex'
        
        if not main_file.exists():
            print(f"Error: {main_file} not found")
            return False
        
        # Change to target directory
        original_dir = os.getcwd()
        os.chdir(target_dir)
        
        try:
            # Clean auxiliary files if requested
            if clean:
                self._clean_aux_files(target_dir)
            
            # Run LaTeX compilation
            engine = self.config['latex']['engine']
            options = self.config['latex']['options']
            
            # First pass
            cmd = [engine] + options + ['main.tex']
            subprocess.run(cmd, check=True)
            
            # Run BibTeX if bibliography exists
            if (target_dir / 'src' / 'bibliography.bib').exists():
                subprocess.run([self.config['latex']['bibtex'], 'main'], check=True)
                # Two more passes for citations
                subprocess.run(cmd, check=True)
                subprocess.run(cmd, check=True)
            
            # Final pass
            subprocess.run(cmd, check=True)
            
            # Move PDF to versioned output if enabled
            if self.config['output']['versioning']:
                self._version_output(target_dir / 'main.pdf', target)
            
            print(f"✓ Compilation successful!")
            
            # Open PDF if requested
            if view:
                self._open_pdf(target_dir / 'main.pdf')
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"✗ Compilation failed: {e}")
            return False
        finally:
            os.chdir(original_dir)
    
    def _clean_aux_files(self, directory: Path):
        """Clean auxiliary LaTeX files."""
        aux_extensions = ['.aux', '.log', '.out', '.toc', '.bbl', '.blg', 
                         '.fls', '.fdb_latexmk', '.synctex.gz']
        for ext in aux_extensions:
            for file in directory.glob(f'*{ext}'):
                file.unlink()
    
    def _version_output(self, pdf_path: Path, target: str):
        """Create versioned output."""
        if not pdf_path.exists():
            return
        
        build_dir = self.root_dir / self.config['directories']['build']
        build_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        version = self.config['project']['version']
        
        if self.config['output']['timestamp']:
            output_name = f"{target}_v{version}_{timestamp}.pdf"
        else:
            output_name = f"{target}_v{version}.pdf"
        
        shutil.copy2(pdf_path, build_dir / output_name)
        
        # Also create a 'latest' symlink
        latest_link = build_dir / f"{target}_latest.pdf"
        if latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(output_name)
    
    def _open_pdf(self, pdf_path: Path):
        """Open PDF with system viewer."""
        if sys.platform == 'darwin':
            subprocess.run(['open', str(pdf_path)])
        elif sys.platform == 'linux':
            subprocess.run(['xdg-open', str(pdf_path)])
        elif sys.platform == 'win32':
            subprocess.run(['start', str(pdf_path)], shell=True)
    
    def create_section(self, section: str, target: str = 'manuscript'):
        """Create a new section file from template."""
        template_dir = self.root_dir / self.config['directories']['templates']
        target_dir = self.root_dir / self.config['directories'][target] / 'src'
        
        template_file = template_dir / f"{section}_template.tex"
        output_file = target_dir / f"{section}.tex"
        
        if output_file.exists():
            print(f"Warning: {output_file} already exists")
            return
        
        if template_file.exists():
            shutil.copy2(template_file, output_file)
        else:
            # Create basic template
            content = f"""% {section.title()}
% Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

\\section{{{section.title()}}}

% Add your content here

"""
            output_file.write_text(content)
        
        print(f"✓ Created {output_file}")
    
    def word_count(self, target: str = 'manuscript'):
        """Count words in document."""
        target_dir = self.root_dir / self.config['directories'][target]
        
        # Use texcount if available
        try:
            result = subprocess.run(
                ['texcount', '-brief', str(target_dir / 'main.tex')],
                capture_output=True, text=True
            )
            print(result.stdout)
        except FileNotFoundError:
            print("texcount not found. Install it for word counting.")
    
    def diff(self, old_version: str, new_version: str = 'current'):
        """Create diff between versions."""
        # Implementation for latexdiff
        pass


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='SciTeX - Modern Scientific Paper Management'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Compile command
    compile_parser = subparsers.add_parser('compile', help='Compile document')
    compile_parser.add_argument('target', nargs='?', default='manuscript',
                               choices=['manuscript', 'revision', 'supplementary'])
    compile_parser.add_argument('-c', '--clean', action='store_true',
                               help='Clean auxiliary files')
    compile_parser.add_argument('-v', '--view', action='store_true',
                               help='Open PDF after compilation')
    
    # Create command
    create_parser = subparsers.add_parser('create', help='Create new section')
    create_parser.add_argument('section', help='Section name')
    create_parser.add_argument('-t', '--target', default='manuscript',
                              help='Target document')
    
    # Word count command
    wc_parser = subparsers.add_parser('wordcount', help='Count words')
    wc_parser.add_argument('target', nargs='?', default='manuscript')
    
    # Init command
    init_parser = subparsers.add_parser('init', help='Initialize new paper')
    
    args = parser.parse_args()
    
    # Initialize SciTeX
    scitex = SciTeX()
    
    if args.command == 'compile':
        scitex.compile(args.target, clean=args.clean, view=args.view)
    elif args.command == 'create':
        scitex.create_section(args.section, args.target)
    elif args.command == 'wordcount':
        scitex.word_count(args.target)
    elif args.command == 'init':
        print("✓ SciTeX initialized")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()