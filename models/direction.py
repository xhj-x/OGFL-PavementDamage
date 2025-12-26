# Ultralytics YOLO 🚀, AGPL-3.0 license

"""Direction-aware feature enhancement module for road crack detection."""

import torch
import torch.nn as nn


class DirectionFeatureEnhancement(nn.Module):
    """
    Direction-aware Feature Enhancement Module for improving detection of linear cracks.

    This module enhances features in horizontal and vertical directions using direction-sensitive
    convolutions, which helps in better detecting transverse and longitudinal cracks on road surfaces.
    """

    def __init__(self, in_channels, reduction_ratio=4):
        """
        Initialize the Direction Feature Enhancement module.

        Args:
            in_channels (int): Number of input channels
            reduction_ratio (int): Channel reduction ratio for efficiency
        """
        super().__init__()

        # Reduced channels for efficient computation
        reduced_channels = max(8, in_channels // reduction_ratio)

        # Horizontal direction-sensitive branch (for transverse cracks)
        self.horizontal_branch = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=(1, 5), padding=(0, 2), bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.SiLU(),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )

        # Vertical direction-sensitive branch (for longitudinal cracks)
        self.vertical_branch = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=(5, 1), padding=(2, 0), bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.SiLU(),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )

        # Attention mechanism for adaptive feature fusion
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels * 2, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels * 2, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize the weights of the convolution layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass of the direction enhancement module.

        Args:
            x (torch.Tensor): Input feature map

        Returns:
            torch.Tensor: Enhanced feature map with direction-aware features
        """
        # Process through directional branches
        h_features = self.horizontal_branch(x)
        v_features = self.vertical_branch(x)

        # Concatenate features for attention
        combined = torch.cat([h_features, v_features], dim=1)

        # Generate attention weights
        attention_weights = self.attention(combined)

        # Split attention weights for horizontal and vertical features
        h_weights, v_weights = torch.chunk(attention_weights, 2, dim=1)

        # Apply attention and add residual connection
        enhanced = h_weights * h_features + v_weights * v_features + x

        return enhanced


class DFE (nn.Module):
    """
    C2f block with Direction-aware Feature Enhancement.
    This combines the standard C2f block with direction-aware features,
    making it suitable as a drop-in replacement in the YOLOv8 backbone.
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """
        Initialize C2f block with Direction-aware Feature Enhancement.

        Args:
            c1 (int): Input channels
            c2 (int): Output channels
            n (int): Number of C2f blocks
            shortcut (bool): Use shortcut connection
            g (int): Groups
            e (float): Expansion ratio
        """
        super().__init__()
        from ultralytics.nn.modules.block import C2f

        # Create standard C2f block
        self.c2f = C2f(c1, c2, n, shortcut, g, e)

        # Add direction enhancement module after C2f
        self.direction_enhance = DirectionFeatureEnhancement(c2)

    def forward(self, x):
        """Forward pass with direction enhancement."""
        return self.direction_enhance(self.c2f(x))