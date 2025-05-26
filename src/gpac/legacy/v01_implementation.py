"""
V01 implementation of BandPassFilter using depthwise convolution.

This is a legacy implementation preserved for research compatibility.
It uses a simpler depthwise convolution approach that was found to have
better correlation with TensorPAC in some cases.

WARNING: This is not part of the main gPAC API and may be removed.
"""

import torch
import torch.nn as nn
from typing import Optional


class V01BandPassFilter(nn.Module):
    """
    V01 implementation using depthwise convolution for filtering.
    
    This implementation uses a simpler approach with depthwise convolution
    that processes each filter independently. While simpler, it was found
    to have better correlation with TensorPAC in certain scenarios.
    """
    
    def __init__(
        self,
        kernels: torch.Tensor,
        filtfilt_mode: bool = False,
        edge_mode: Optional[str] = None,
    ):
        """
        Initialize V01 filter.
        
        Parameters
        ----------
        kernels : torch.Tensor
            Pre-computed filter kernels
        filtfilt_mode : bool
            Whether to use forward-backward filtering
        edge_mode : str or None
            Edge padding mode
        """
        super().__init__()
        self.register_buffer("kernels", kernels)
        self.filtfilt_mode = filtfilt_mode
        self.edge_mode = edge_mode
        
        # Calculate padlen for edge handling if requested
        if edge_mode:
            self.padlen = kernels.shape[1] - 1
        else:
            self.padlen = 0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply V01 filtering using depthwise convolution."""
        # x shape: (batch*channel*segment, 1, time)
        
        # Apply edge padding if requested
        if self.edge_mode and self.padlen > 0:
            x = torch.nn.functional.pad(
                x, (self.padlen, self.padlen), mode=self.edge_mode
            )
        
        # Expand input to match number of filters
        x_expanded = x.expand(-1, len(self.kernels), -1)
        
        # Prepare kernels for depthwise conv
        kernels_expanded = self.kernels.unsqueeze(1)
        
        if self.filtfilt_mode:
            # First forward pass using depthwise convolution
            filtered = torch.nn.functional.conv1d(
                x_expanded,
                kernels_expanded,
                padding="same",
                groups=len(self.kernels),  # Each filter processes its own channel
            )
            
            # Second pass on time-reversed signal (backward filtering)
            filtered = torch.nn.functional.conv1d(
                filtered.flip(-1),  # Flip time dimension
                kernels_expanded,
                padding="same",
                groups=len(self.kernels),
            ).flip(-1)  # Flip back
        else:
            # Single-pass filtering
            filtered = torch.nn.functional.conv1d(
                x_expanded,
                kernels_expanded,
                padding="same",
                groups=len(self.kernels),
            )
        
        # Remove edge padding if it was applied
        if self.edge_mode and self.padlen > 0:
            filtered = filtered[:, :, self.padlen : -self.padlen]
        
        # Add extra dimension to match expected output
        filtered = filtered.unsqueeze(1)
        
        return filtered