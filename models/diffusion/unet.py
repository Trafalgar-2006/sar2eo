"""
models/diffusion/unet.py — Conditional Denoising U-Net

Architecture for SAR-conditioned image diffusion:
  - SAR image is concatenated as additional channels to the noisy EO
  - Time embedding (sinusoidal → MLP) is injected into every ResNet block
  - Self-attention at 16×16 and 8×8 for global consistency
  - ~20M parameters — fits easily alongside the GAN in memory

Input:
  x_noisy : [B, 3, 256, 256]   noisy EO image at timestep t
  sar     : [B, 1, 256, 256]   SAR condition
  t       : [B]                integer timestep (0..T-1)

Output:
  [B, 3, 256, 256]  predicted noise ε (or x_0 in x0-prediction mode)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Time embedding
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal positional encoding for timestep t.
    dim must be divisible by 2.
    """
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B]
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )                                         # [half]
        args  = t[:, None].float() * freqs[None]  # [B, half]
        return torch.cat([args.sin(), args.cos()], dim=-1)  # [B, dim]


class TimeEmbedMLP(nn.Module):
    """Project sinusoidal embedding to model width."""
    def __init__(self, time_dim: int, model_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(time_dim, model_dim * 4),
            nn.SiLU(),
            nn.Linear(model_dim * 4, model_dim),
        )

    def forward(self, t_emb: torch.Tensor) -> torch.Tensor:
        return self.net(t_emb)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class GroupNormAct(nn.Sequential):
    def __init__(self, channels: int, groups: int = 32):
        # Use min(groups, channels) to avoid error when channels < 32
        g = min(groups, channels)
        while channels % g != 0:
            g //= 2
        super().__init__(nn.GroupNorm(g, channels), nn.SiLU())


class ResBlock(nn.Module):
    """
    ResNet block with time embedding injection.
    Optionally doubles as a Dropout block (used in deeper stages).
    """
    def __init__(self, in_ch: int, out_ch: int, time_dim: int,
                 dropout: float = 0.0):
        super().__init__()
        self.norm1  = GroupNormAct(in_ch)
        self.conv1  = nn.Conv2d(in_ch,  out_ch, 3, padding=1)
        self.t_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))
        self.norm2  = GroupNormAct(out_ch)
        self.drop   = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2  = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip   = (nn.Conv2d(in_ch, out_ch, 1)
                       if in_ch != out_ch else nn.Identity())

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.norm1(x))
        h = h + self.t_proj(t_emb)[:, :, None, None]
        h = self.conv2(self.drop(self.norm2(h)))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    """Multi-head self-attention for spatial feature maps (flattened)."""
    def __init__(self, channels: int, n_heads: int = 8):
        super().__init__()
        assert channels % n_heads == 0
        self.norm  = nn.GroupNorm(1, channels)
        self.attn  = nn.MultiheadAttention(channels, n_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).flatten(2).permute(0, 2, 1)   # [B, H*W, C]
        h, _ = self.attn(h, h, h)
        return x + h.permute(0, 2, 1).view(B, C, H, W)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


# ---------------------------------------------------------------------------
# Conditional U-Net
# ---------------------------------------------------------------------------

class ConditionalUNet(nn.Module):
    """
    Conditional denoising U-Net for SAR-to-EO diffusion.

    SAR condition is concatenated to the noisy EO in channel dim.
    Time step is injected via sinusoidal embedding into every ResBlock.

    Architecture:
        256×256 : [4ch → 64ch]  ResBlock × 2
        128×128 : [64 → 128ch]  ResBlock × 2
         64×64  : [128 → 256ch] ResBlock × 2
         32×32  : [256 → 256ch] ResBlock × 2
         16×16  : [256 → 256ch] ResBlock × 2 + Self-Attention
          8×8   : [256 → 256ch] ResBlock × 2 + Self-Attention  (bottleneck)
        ↑ decoder mirrors encoder with skip connections
    """

    def __init__(
        self,
        in_channels:  int   = 4,    # 3 (noisy EO) + 1 (SAR)
        out_channels: int   = 3,    # predicted noise (EO channels)
        base_ch:      int   = 64,
        ch_mult:      tuple = (1, 2, 4, 4, 4, 4),   # 6 levels
        n_res_blocks: int   = 2,
        attn_levels:  tuple = (4, 5),                # levels with self-attention
        dropout:      float = 0.1,
        time_dim:     int   = 256,
    ):
        super().__init__()

        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp   = TimeEmbedMLP(time_dim, time_dim)

        # Initial projection
        self.in_conv = nn.Conv2d(in_channels, base_ch, 3, padding=1)

        # Channel sizes per level
        chs = [base_ch * m for m in ch_mult]

        # ── Encoder ──────────────────────────────────────────────────────────
        self.down_blocks  = nn.ModuleList()
        self.down_samples = nn.ModuleList()

        in_ch = base_ch
        for i, ch in enumerate(chs):
            level_blocks = nn.ModuleList()
            for _ in range(n_res_blocks):
                level_blocks.append(ResBlock(in_ch, ch, time_dim, dropout))
                if i in attn_levels:
                    level_blocks.append(SelfAttention(ch))
                in_ch = ch
            self.down_blocks.append(level_blocks)
            if i < len(chs) - 1:
                self.down_samples.append(Downsample(ch))
            else:
                self.down_samples.append(nn.Identity())

        # ── Bottleneck ────────────────────────────────────────────────────────
        self.mid_res1 = ResBlock(in_ch, in_ch, time_dim, dropout)
        self.mid_attn = SelfAttention(in_ch)
        self.mid_res2 = ResBlock(in_ch, in_ch, time_dim, dropout)

        # ── Decoder ──────────────────────────────────────────────────────────
        self.up_blocks   = nn.ModuleList()
        self.up_samples  = nn.ModuleList()

        for i, ch in reversed(list(enumerate(chs))):
            skip_ch  = ch
            level_blocks = nn.ModuleList()
            for j in range(n_res_blocks + 1):   # +1 extra block to handle skip cat
                # First block: take skip cat → in_ch + skip_ch
                block_in = in_ch + skip_ch if j == 0 else ch
                level_blocks.append(ResBlock(block_in, ch, time_dim, dropout))
                if i in attn_levels:
                    level_blocks.append(SelfAttention(ch))
                in_ch = ch
            self.up_blocks.append(level_blocks)
            if i > 0:
                self.up_samples.append(Upsample(ch))
            else:
                self.up_samples.append(nn.Identity())

        # Output
        self.out_norm = GroupNormAct(in_ch)
        self.out_conv = nn.Conv2d(in_ch, out_channels, 3, padding=1)

        # Init
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def encoder_parameters(self):
        """Parameters in encoder path (for differential LR if needed)."""
        return list(self.down_blocks.parameters()) + list(self.in_conv.parameters())

    def forward(self,
                x_noisy: torch.Tensor,
                sar:     torch.Tensor,
                t:       torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_noisy : [B, 3, H, W]  noisy EO at timestep t
            sar     : [B, 1, H, W]  SAR condition
            t       : [B]           integer timestep

        Returns:
            [B, 3, H, W]  predicted noise (or x_0 depending on objective)
        """
        # Concatenate SAR as extra channel
        x = torch.cat([x_noisy, sar], dim=1)   # [B, 4, H, W]

        # Time embedding
        t_emb = self.time_mlp(self.time_embed(t))  # [B, time_dim]

        # Initial conv
        h = self.in_conv(x)

        # ── Encoder ──────────────────────────────────────────────────────────
        skips = []
        for blocks, down in zip(self.down_blocks, self.down_samples):
            for block in blocks:
                if isinstance(block, ResBlock):
                    h = block(h, t_emb)
                else:
                    h = block(h)
            skips.append(h)
            h = down(h)

        # ── Bottleneck ────────────────────────────────────────────────────────
        h = self.mid_res1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_res2(h, t_emb)

        # ── Decoder ──────────────────────────────────────────────────────────
        for blocks, up in zip(self.up_blocks, self.up_samples):
            skip = skips.pop()
            first = True
            for block in blocks:
                if isinstance(block, ResBlock):
                    if first:
                        h = block(torch.cat([h, skip], dim=1), t_emb)
                        first = False
                    else:
                        h = block(h, t_emb)
                else:
                    h = block(h)
            h = up(h)

        return self.out_conv(self.out_norm(h))


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = ConditionalUNet()
    n = sum(p.numel() for p in model.parameters())
    print(f"ConditionalUNet params: {n:,}")

    B = 2
    x = torch.randn(B, 3, 256, 256)
    s = torch.randn(B, 1, 256, 256)
    t = torch.randint(0, 1000, (B,))
    out = model(x, s, t)
    print(f"Output: {out.shape}  range=[{out.min():.2f},{out.max():.2f}]")
