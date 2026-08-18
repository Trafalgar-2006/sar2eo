"""
discriminator.py — Multi-Scale PatchGAN Discriminator

Replaces the single 70×70 PatchGAN with 3 discriminators operating at
different input resolutions: 1×, 0.5×, and 0.25× of the original image.

Why multi-scale:
  A single PatchGAN has a fixed receptive field and thus only sees texture
  at one scale. Urban SAR features (buildings, roads, infrastructure) have
  structure at MULTIPLE spatial scales simultaneously:
    - Fine detail (roads, small buildings): needs high-resolution D
    - Coarse layout (city blocks, urban density): needs low-resolution D
  With a single-scale D, the generator can "fool" it by getting the fine
  texture right while having wrong coarse structure (or vice versa).
  The multi-scale discriminator forces the generator to be realistic at all
  scales simultaneously.

  Reference: Wang et al. (2018). High-Resolution Image Synthesis and
  Semantic Manipulation with Conditional GANs (pix2pixHD).
  https://arxiv.org/abs/1711.11585

Architecture:
  D_0: operates on full (256×256) input
  D_1: operates on 2× downsampled (128×128) input
  D_2: operates on 4× downsampled (64×64) input

  Each D_i is an independent 70×70 PatchGAN (n_layers=3).
  The discriminator forward pass returns a LIST of patch maps.
  The GAN loss averages across all 3 scales.

Input per scale:  [B, 1+3, H, W] — concatenated (SAR, EO)
Output per scale: [B, 1, ~H/16, ~W/16] — raw logits (no sigmoid)
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Single-scale 70×70 PatchGAN (same as original, unchanged)
# ---------------------------------------------------------------------------

class PatchGANDiscriminator(nn.Module):
    """
    Single-scale 70×70 PatchGAN discriminator (Pix2Pix architecture).
    Used as a building block for the multi-scale discriminator.

    Args:
        in_channels  (int): SAR channels (1)
        out_channels (int): EO channels (3) — concatenated with SAR as input
        base_ch      (int): Base channel count (NDF = 64)
        n_layers     (int): Number of conv layers with stride=2 (default 3)
    """

    def __init__(self, in_channels: int = 1,
                 out_channels: int = 3,
                 base_ch: int = 64,
                 n_layers: int = 3):
        super().__init__()
        ndf      = base_ch
        input_nc = in_channels + out_channels   # 1 + 3 = 4

        layers = []

        # First layer — no BatchNorm
        layers += [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Intermediate layers — stride=2, growing channels
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult      = min(2 ** n, 8)
            layers += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        # Penultimate layer — stride=1
        nf_mult_prev = nf_mult
        nf_mult      = min(2 ** n_layers, 8)
        layers += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Final layer — stride=1, no BN, no activation
        layers += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, sar: torch.Tensor, eo: torch.Tensor) -> torch.Tensor:
        x = torch.cat([sar, eo], dim=1)
        return self.model(x)

    def init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)


# ---------------------------------------------------------------------------
# Multi-Scale Discriminator
# ---------------------------------------------------------------------------

class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale discriminator: 3 PatchGAN discriminators at different scales.

    Forward pass returns a list of patch prediction maps, one per scale.
    The GANLoss class handles averaging the loss across scales.

    Args:
        in_channels  (int): SAR channels (1)
        out_channels (int): EO channels (3)
        base_ch      (int): Base channel count per discriminator (64)
        n_layers     (int): PatchGAN depth per discriminator (3)
        n_scales     (int): Number of scales (default 3)
    """

    def __init__(self, in_channels: int = 1,
                 out_channels: int = 3,
                 base_ch: int = 64,
                 n_layers: int = 3,
                 n_scales: int = 3):
        super().__init__()
        self.n_scales = n_scales

        # One independent discriminator per scale
        self.discriminators = nn.ModuleList([
            PatchGANDiscriminator(in_channels, out_channels, base_ch, n_layers)
            for _ in range(n_scales)
        ])

        # Downsampler for scale reduction between discriminators
        # count_include_pad=False avoids zero-padding bias at image boundaries
        self.downsample = nn.AvgPool2d(
            kernel_size=3, stride=2, padding=1, count_include_pad=False
        )

    def forward(self, sar: torch.Tensor,
                eo: torch.Tensor) -> list:
        """
        Args:
            sar: [B, 1, H, W]  — SAR input (condition)
            eo:  [B, 3, H, W]  — EO image (real or generated)
        Returns:
            List of n_scales patch prediction tensors, one per scale.
            Each is [B, 1, ~H/16, ~W/16] raw logits (no sigmoid).
        """
        outputs = []
        for i, D in enumerate(self.discriminators):
            if i > 0:
                sar = self.downsample(sar)
                eo  = self.downsample(eo)
            outputs.append(D(sar, eo))
        return outputs

    def init_weights(self) -> None:
        for D in self.discriminators:
            D.init_weights()


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Windows consoles default to cp1252 and raise on the unicode used in the
    # progress output below. Force UTF-8 so local runs match Kaggle/Linux.
    import sys as _sys
    try:
        _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    D = MultiScaleDiscriminator(
        in_channels=1, out_channels=3, base_ch=64, n_layers=3, n_scales=3
    )
    D.init_weights()

    sar = torch.randn(2, 1, 256, 256)
    eo  = torch.randn(2, 3, 256, 256)
    out = D(sar, eo)

    print("Multi-scale discriminator outputs:")
    for i, o in enumerate(out):
        print(f"  Scale {i} (input {256 // (2**i)}×{256 // (2**i)}): "
              f"output {tuple(o.shape)}")

    total_params = sum(p.numel() for p in D.parameters() if p.requires_grad)
    print(f"Total discriminator params: {total_params:,}")
    print("Discriminator OK. ✓")
