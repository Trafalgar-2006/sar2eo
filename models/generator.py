"""
generator.py — U-Net Generator for SAR-to-EO Translation

Architecture:
  - Encoder–decoder with skip connections (U-Net style)
  - 8 downsampling levels for 256×256 input → 1×1 bottleneck
  - base_channels=64 (fits 4GB VRAM at batch=4 with fp16)
  - Input:  [B, 1, 256, 256]  — single-channel SAR (VV)
  - Output: [B, 3, 256, 256]  — RGB EO image, tanh activation → [-1, 1]

Channel progression (enc → dec with skip):
  Encoder: 1→64→128→256→512→512→512→512→512
  Decoder: 512→512→512→512→512→256→128→64→3

Encoder blocks: Conv2d(stride=2) → BatchNorm → LeakyReLU(0.2)
Decoder blocks: ConvTranspose2d(stride=2) → BatchNorm → ReLU (+ dropout on deepest 3)
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class EncoderBlock(nn.Module):
    """
    One encoder step: Conv(stride=2) → [BatchNorm] → LeakyReLU
    No BN on the first layer (direct input).
    """
    def __init__(self, in_ch: int, out_ch: int, use_bn: bool = True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=not use_bn)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    """
    One decoder step: ConvTranspose(stride=2) → BatchNorm → ReLU → [Dropout]
    Optionally applies dropout (on the 3 deepest decoder layers as per Pix2Pix).
    """
    def __init__(self, in_ch: int, out_ch: int,
                 dropout: bool = False, use_bn: bool = True):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=not use_bn),
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# U-Net Generator
# ---------------------------------------------------------------------------

class UNetGenerator(nn.Module):
    """
    U-Net generator following the Pix2Pix architecture.

    Args:
        in_channels  (int): Number of input channels  (1 for SAR VV)
        out_channels (int): Number of output channels (3 for RGB EO)
        base_ch      (int): Base channel count. 64 fits 4GB VRAM @ batch=4 fp16.
    """

    def __init__(self, in_channels: int = 1,
                 out_channels: int = 3,
                 base_ch: int = 64):
        super().__init__()
        ngf = base_ch

        # ------------------------------------------------------------------
        # Encoder (8 downsampling steps: 256 → 1)
        # ------------------------------------------------------------------
        # E1: no BatchNorm on first layer
        self.enc1 = EncoderBlock(in_channels, ngf,       use_bn=False)  # 256→128
        self.enc2 = EncoderBlock(ngf,         ngf * 2)                  # 128→64
        self.enc3 = EncoderBlock(ngf * 2,     ngf * 4)                  # 64→32
        self.enc4 = EncoderBlock(ngf * 4,     ngf * 8)                  # 32→16
        self.enc5 = EncoderBlock(ngf * 8,     ngf * 8)                  # 16→8
        self.enc6 = EncoderBlock(ngf * 8,     ngf * 8)                  # 8→4
        self.enc7 = EncoderBlock(ngf * 8,     ngf * 8)                  # 4→2
        self.enc8 = EncoderBlock(ngf * 8,     ngf * 8, use_bn=False)    # 2→1 (bottleneck, no BN)

        # ------------------------------------------------------------------
        # Decoder (8 upsampling steps: 1 → 256)
        # Skip connections double the input channels (cat with encoder output)
        # ------------------------------------------------------------------
        self.dec8 = DecoderBlock(ngf * 8,         ngf * 8, dropout=True)   # 1→2,   in=512
        self.dec7 = DecoderBlock(ngf * 8 * 2,     ngf * 8, dropout=True)   # 2→4,   in=1024 (cat enc7)
        self.dec6 = DecoderBlock(ngf * 8 * 2,     ngf * 8, dropout=True)   # 4→8,   in=1024 (cat enc6)
        self.dec5 = DecoderBlock(ngf * 8 * 2,     ngf * 8)                 # 8→16,  in=1024 (cat enc5)
        self.dec4 = DecoderBlock(ngf * 8 * 2,     ngf * 4)                 # 16→32, in=1024 (cat enc4)
        self.dec3 = DecoderBlock(ngf * 4 * 2,     ngf * 2)                 # 32→64, in=512  (cat enc3)
        self.dec2 = DecoderBlock(ngf * 2 * 2,     ngf)                     # 64→128,in=256  (cat enc2)

        # Final layer: no BN, tanh output
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, out_channels,   # 128→256, in=128 (cat enc1)
                               kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        e8 = self.enc8(e7)

        # Decode with skip connections (concatenate along channel dim)
        d8 = self.dec8(e8)
        d7 = self.dec7(torch.cat([d8, e7], dim=1))
        d6 = self.dec6(torch.cat([d7, e6], dim=1))
        d5 = self.dec5(torch.cat([d6, e5], dim=1))
        d4 = self.dec4(torch.cat([d5, e4], dim=1))
        d3 = self.dec3(torch.cat([d4, e3], dim=1))
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        d1 = self.dec1(torch.cat([d2, e1], dim=1))

        return d1   # [B, 3, 256, 256], range [-1, 1]

    def init_weights(self):
        """Initialise weights with N(0, 0.02) as per Pix2Pix paper."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
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
    G = UNetGenerator(in_channels=1, out_channels=3, base_ch=64)
    G.init_weights()
    x = torch.randn(2, 1, 256, 256)
    out = G(x)
    print(f"Input shape : {x.shape}")
    print(f"Output shape: {out.shape}")   # expect [2, 3, 256, 256]
    print(f"Output range: [{out.min():.3f}, {out.max():.3f}]")

    n_params = sum(p.numel() for p in G.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params:,}")
