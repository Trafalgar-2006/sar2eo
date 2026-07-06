"""
attention.py — CBAM: Convolutional Block Attention Module

Reference: Woo et al. (2018). CBAM: Convolutional Block Attention Module. ECCV 2018.
           https://arxiv.org/abs/1807.06521

CBAM applies two sequential attention gates to a feature map:
  1. Channel Attention  — "what" to focus on  (which features matter)
  2. Spatial Attention  — "where" to focus on (which locations matter)

Physical motivation for SAR-to-EO:
  - Channel attention: SAR and EO have very different spectral statistics.
    Not all ResNet features trained on natural images are useful for SAR.
    Channel attention suppresses irrelevant filters and amplifies useful ones.
  - Spatial attention: SAR speckle is spatially non-uniform. Urban areas have
    coherent backscatter, vegetation is diffuse, water is near-zero. The model
    can learn to selectively trust encoder features by location.

Usage in U-Net skip connections:
    cbam = CBAM(in_channels)
    attended_skip = cbam(encoder_feature_map)
    decoder_input = torch.cat([upsample, attended_skip], dim=1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Channel attention: generates a per-channel weight vector from both
    average-pooled and max-pooled global descriptors, passed through a
    shared MLP.

    Args:
        in_ch      (int): Number of input channels.
        reduction  (int): Channel reduction ratio for the MLP. Default 16.
    """

    def __init__(self, in_ch: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, in_ch // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # Shared MLP (implemented as 1×1 convs for efficiency)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_ch, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            Channel-attended x: [B, C, H, W]
        """
        avg_out = self.mlp(self.avg_pool(x))    # [B, C, 1, 1]
        max_out = self.mlp(self.max_pool(x))    # [B, C, 1, 1]
        scale   = self.sigmoid(avg_out + max_out)
        return x * scale


class SpatialAttention(nn.Module):
    """
    Spatial attention: generates a per-location weight map from
    channel-pooled descriptors, passed through a 7×7 conv.

    Args:
        kernel_size (int): Conv kernel size. 7 recommended by the paper.
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            Spatially-attended x: [B, C, H, W]
        """
        avg_out = x.mean(dim=1, keepdim=True)      # [B, 1, H, W]
        max_out = x.max(dim=1, keepdim=True)[0]    # [B, 1, H, W]
        pooled  = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
        scale   = self.sigmoid(self.conv(pooled))   # [B, 1, H, W]
        return x * scale


class CBAM(nn.Module):
    """
    CBAM: Channel attention followed by Spatial attention.

    Args:
        in_ch      (int): Input channels.
        reduction  (int): Channel reduction ratio (default 16).
        kernel_size(int): Spatial attention kernel size (default 7).
    """

    def __init__(self, in_ch: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel = ChannelAttention(in_ch, reduction)
        self.spatial = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] encoder feature map
        Returns:
            Attended feature map [B, C, H, W] — same shape, refined
        """
        x = self.channel(x)
        x = self.spatial(x)
        return x


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for ch in [64, 128, 256, 512]:
        cbam = CBAM(ch)
        x = torch.randn(2, ch, 32, 32)
        out = cbam(x)
        assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"
        n_params = sum(p.numel() for p in cbam.parameters())
        print(f"CBAM({ch:4d}ch): output={out.shape}, params={n_params:,}")
    print("CBAM OK.")
