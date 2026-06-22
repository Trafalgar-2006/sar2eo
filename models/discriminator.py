"""
discriminator.py — PatchGAN 70×70 Discriminator

Architecture:
  - Classifies overlapping 70×70 patches of the image as real/fake
  - Input: concatenated (SAR, EO) — shape [B, in_ch+out_ch, 256, 256]
  - Output: patch probability map — shape [B, 1, 30, 30]
  - Uses BCEWithLogitsLoss (no sigmoid in forward pass)

Layer structure (Pix2Pix standard, n_layers=3):
  Conv → [BN] → LReLU   (no BN on first layer)
  Conv → BN → LReLU
  Conv → BN → LReLU
  Conv → BN → LReLU     (stride=1 padding=1 — penultimate)
  Conv                   (stride=1 padding=1 — final, no BN, no activation)
"""

import torch
import torch.nn as nn


class PatchGANDiscriminator(nn.Module):
    """
    70×70 PatchGAN discriminator from Pix2Pix.

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
        ndf = base_ch
        # Discriminator takes concatenated (SAR, EO) as input
        input_nc = in_channels + out_channels   # 1 + 3 = 4

        layers = []

        # First layer — no BatchNorm
        layers += [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Intermediate layers with stride=2 and growing channels
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        # Penultimate layer — stride=1
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        layers += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Final layer — stride=1, no BN, no activation
        # Output: patch prediction map
        layers += [
            nn.Conv2d(ndf * nf_mult, 1,
                      kernel_size=4, stride=1, padding=1),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, sar: torch.Tensor, eo: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sar: [B, 1, H, W]  — SAR input (condition)
            eo:  [B, 3, H, W]  — EO image (real or generated)
        Returns:
            patch_pred: [B, 1, ~30, ~30]  — raw logits (no sigmoid)
        """
        x = torch.cat([sar, eo], dim=1)   # [B, 4, H, W]
        return self.model(x)

    def init_weights(self):
        """Initialise weights with N(0, 0.02) as per Pix2Pix paper."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    D = PatchGANDiscriminator(in_channels=1, out_channels=3, base_ch=64, n_layers=3)
    D.init_weights()

    sar = torch.randn(2, 1, 256, 256)
    eo  = torch.randn(2, 3, 256, 256)
    out = D(sar, eo)

    print(f"SAR shape   : {sar.shape}")
    print(f"EO shape    : {eo.shape}")
    print(f"Output shape: {out.shape}")   # expect [2, 1, 30, 30]

    n_params = sum(p.numel() for p in D.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params:,}")
