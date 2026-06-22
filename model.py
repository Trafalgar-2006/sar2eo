"""
model.py — GalaxEye Assignment submission shim

The actual model implementation is split across:
  models/generator.py    — UNetGenerator (U-Net with skip connections)
  models/discriminator.py — PatchGANDiscriminator (70x70 PatchGAN)
  models/losses.py       — L1Loss, GANLoss, FFTLoss, VGGPerceptualLoss

This file re-exports both models from the top level to satisfy the
assignment deliverable naming convention (model.py at repo root).

Usage:
    from model import UNetGenerator, PatchGANDiscriminator
"""

from models.generator import UNetGenerator
from models.discriminator import PatchGANDiscriminator
from models.losses import L1Loss, GANLoss, FFTLoss, VGGPerceptualLoss

__all__ = [
    "UNetGenerator",
    "PatchGANDiscriminator",
    "L1Loss",
    "GANLoss",
    "FFTLoss",
    "VGGPerceptualLoss",
]
