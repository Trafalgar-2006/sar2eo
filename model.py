"""
model.py — Root-level shim

Re-exports the main model classes for convenience. Allows:
    from model import UNetGenerator, MultiScaleDiscriminator
instead of navigating into models/.
"""

from models.generator     import UNetGenerator
from models.discriminator import MultiScaleDiscriminator, PatchGANDiscriminator
from models.losses        import (
    L1Loss, GANLoss, FFTLoss, VGGPerceptualLoss, MSSSIMLoss
)
from models.attention     import CBAM, ChannelAttention, SpatialAttention

__all__ = [
    "UNetGenerator",
    "MultiScaleDiscriminator",
    "PatchGANDiscriminator",
    "L1Loss", "GANLoss", "FFTLoss", "VGGPerceptualLoss", "MSSSIMLoss",
    "CBAM", "ChannelAttention", "SpatialAttention",
]
