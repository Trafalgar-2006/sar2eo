"""
train_diffusion_ldm.py — Phase 2: Latent Diffusion Model (LDM) for SAR2EO

KEY UPGRADE over train_diffusion.py:
  - Diffusion runs in LATENT SPACE (32×32×4) not pixel space (256×256×3)
  - Uses frozen pretrained VAE from Stable Diffusion 1.5
  - 64× fewer tokens per training step = much faster + more stable
  - SAR conditioned via cross-attention (not just concatenation)
  - Same DDPM/DDIM framework, just in latent space

WHY THIS IS BETTER THAN PIXEL DIFFUSION:
  Pixel space diffusion (train_diffusion.py):
    Each step: [B, 4, 256, 256] → 262K tokens per batch
    VGG has to run in pixel space → slow
    High-freq details bleed into noise schedule → unstable

  LDM (this file):
    Each step: [B, 4, 32, 32] → 4096 tokens per batch (64× less)
    VAE separately handles pixel reconstruction → denoiser focuses on semantics
    Much faster convergence, better perceptual quality

USAGE:
  # Install diffusers first:
  pip install diffusers transformers accelerate

  python train_diffusion_ldm.py --config config.yaml
  # OR on Kaggle: exec(open("kaggle_train_diffusion.py", encoding="utf-8").read())

EXPECTED: ~8-10hrs on T4, LPIPS ~0.28, FID ~80 (vs GAN ~0.45, ~180)
"""

import os
import sys
import json
import yaml
import random
import argparse
import time
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.amp
from tqdm import tqdm

from data.dataloader import get_dataloaders
from models.diffusion.unet import ConditionalUNet
from models.diffusion.ddpm import DDPM, DDIMSampler


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# VAE wrapper (frozen SD-1.5 VAE OR simple pixel passthrough)
# ---------------------------------------------------------------------------

class LatentEncoder:
    """
    Encodes images to latent space using the SD-1.5 VAE (frozen).
    Falls back to identity (pixel space) if diffusers not installed.

    SD-1.5 VAE: 256×256×3 → 32×32×4 (compression factor 8, 4 latent channels)
    Scaling factor 0.18215 (SD convention — normalises latent to ~unit variance)
    """
    def __init__(self, device: torch.device):
        self.device = device
        self.vae    = None
        self.scale  = 0.18215
        self.latent_channels = 4

        try:
            from diffusers import AutoencoderKL
            print("  Loading SD-1.5 VAE (frozen) ...")
            self.vae = AutoencoderKL.from_pretrained(
                "runwayml/stable-diffusion-v1-5", subfolder="vae"
            ).to(device)
            self.vae.requires_grad_(False)
            self.vae.eval()
            print("  ✓ VAE loaded — diffusion runs in 32×32 latent space")
        except ImportError:
            print("  ⚠ diffusers not found — falling back to pixel space")
            print("    Install with: pip install diffusers")
            self.latent_channels = 3

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 3, 256, 256] in [-1, 1] → latents [B, 4, 32, 32]"""
        if self.vae is None:
            # Pixel fallback: downsample to 32×32 as rough approximation
            return torch.nn.functional.interpolate(x, size=(32, 32), mode="bilinear",
                                                    align_corners=False)
        with torch.no_grad():
            x_01 = (x + 1) / 2  # VAE expects [0, 1]
            latents = self.vae.encode(x_01).latent_dist.sample()
            return latents * self.scale

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: [B, 4, 32, 32] → x [B, 3, 256, 256] in [-1, 1]"""
        if self.vae is None:
            return torch.nn.functional.interpolate(z, size=(256, 256), mode="bilinear",
                                                    align_corners=False)
        with torch.no_grad():
            z_unscaled = z / self.scale
            decoded    = self.vae.decode(z_unscaled).sample
            return decoded.clamp(-1, 1)


class SARLatentEncoder(nn.Module):
    """
    Projects SAR image [B, 1, 256, 256] to latent context [B, latent_h*latent_w, ctx_dim]
    for cross-attention conditioning of the denoising U-Net.

    Architecture:
        Conv stack (256 → 32, stride=8) → flatten spatial → linear projection
    """
    def __init__(self, ctx_dim: int = 512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),   # 256→128
            nn.SiLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 128→64
            nn.SiLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # 64→32
            nn.SiLU(),
        )
        self.proj = nn.Conv2d(128, ctx_dim, 1)  # → [B, ctx_dim, 32, 32]

    def forward(self, sar: torch.Tensor) -> torch.Tensor:
        """
        Returns: [B, 32*32, ctx_dim] — seq of context tokens for cross-attention
        """
        feat = self.encoder(sar)   # [B, 128, 32, 32]
        ctx  = self.proj(feat)     # [B, ctx_dim, 32, 32]
        B, C, H, W = ctx.shape
        return ctx.flatten(2).permute(0, 2, 1)  # [B, H*W, ctx_dim]


# ---------------------------------------------------------------------------
# LDM denoising U-Net (operates in latent space)
# ---------------------------------------------------------------------------

class LDMUNet(nn.Module):
    """
    Lightweight U-Net for denoising in the 32×32 latent space.
    Much smaller than the pixel-space ConditionalUNet — 32×32 vs 256×256.

    Key difference from ConditionalUNet:
        - Operates on 4 latent channels (not 3 pixel channels)
        - No SAR concatenation — SAR is injected via cross-attention context
        - Time embedding via sinusoidal + linear projection (same as DDPM standard)
    """
    def __init__(
        self,
        in_channels:  int = 4,   # latent channels
        out_channels: int = 4,   # predict noise in latent space
        base_ch:      int = 128,
        ctx_dim:      int = 512, # SAR context dimension (from SARLatentEncoder)
        time_dim:     int = 256,
    ):
        super().__init__()
        ch = base_ch

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, ch * 4),
        )
        self._time_dim = time_dim

        # Encoder
        self.enc1 = self._block(in_channels, ch,     ch * 4, ctx_dim)
        self.enc2 = self._block(ch,          ch * 2, ch * 4, ctx_dim)
        self.enc3 = self._block(ch * 2,      ch * 4, ch * 4, ctx_dim)

        self.down1 = nn.Conv2d(ch,     ch,     4, 2, 1)  # 32→16
        self.down2 = nn.Conv2d(ch * 2, ch * 2, 4, 2, 1)  # 16→8
        self.down3 = nn.Conv2d(ch * 4, ch * 4, 4, 2, 1)  # 8→4

        # Bottleneck
        self.mid   = self._block(ch * 4, ch * 4, ch * 4, ctx_dim)

        # Decoder
        self.up3   = nn.ConvTranspose2d(ch * 4, ch * 4, 4, 2, 1)  # 4→8
        self.up2   = nn.ConvTranspose2d(ch * 4, ch * 2, 4, 2, 1)  # 8→16
        self.up1   = nn.ConvTranspose2d(ch * 2, ch,     4, 2, 1)  # 16→32

        self.dec3  = self._block(ch * 8, ch * 4, ch * 4, ctx_dim)
        self.dec2  = self._block(ch * 4, ch * 2, ch * 4, ctx_dim)
        self.dec1  = self._block(ch * 2, ch,     ch * 4, ctx_dim)

        self.out   = nn.Conv2d(ch, out_channels, 1)

    def _block(self, in_ch, out_ch, t_dim, ctx_dim):
        """ResBlock with time + cross-attention conditioning."""
        return ResBlockCrossAttn(in_ch, out_ch, t_dim, ctx_dim)

    @staticmethod
    def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
        half = dim // 2
        freq = torch.exp(
            -torch.arange(half, dtype=torch.float32, device=t.device) *
            (9.21 / (half - 1))  # log(10000) / (half-1)
        )
        args = t[:, None].float() * freq[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                ctx: torch.Tensor) -> torch.Tensor:
        """
        x:   [B, 4, 32, 32]   noisy latents
        t:   [B]              timesteps
        ctx: [B, 1024, 512]   SAR cross-attention context
        Returns: [B, 4, 32, 32] predicted noise
        """
        te = self.sinusoidal_embedding(t, self._time_dim)
        te = self.time_embed(te)  # [B, ch*4]

        e1 = self.enc1(x,             te, ctx)
        e2 = self.enc2(self.down1(e1), te, ctx)
        e3 = self.enc3(self.down2(e2), te, ctx)

        m  = self.mid(self.down3(e3), te, ctx)

        d3 = self.dec3(torch.cat([self.up3(m),  e3], dim=1), te, ctx)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1), te, ctx)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1), te, ctx)

        return self.out(d1)


class ResBlockCrossAttn(nn.Module):
    """ResBlock with time-shift modulation + cross-attention for context."""
    def __init__(self, in_ch, out_ch, t_dim, ctx_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, out_ch * 2)  # scale + shift
        self.act   = nn.SiLU()

        # Cross-attention: latent queries × SAR context keys/values
        self.cross_attn_q = nn.Conv2d(out_ch, out_ch, 1)
        self.cross_attn_k = nn.Linear(ctx_dim, out_ch)
        self.cross_attn_v = nn.Linear(ctx_dim, out_ch)
        self.cross_norm   = nn.GroupNorm(8, out_ch)

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, te, ctx):
        h = self.conv1(self.act(self.norm1(x)))

        # Time-shift modulation
        scale, shift = self.t_proj(self.act(te)).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(self.act(h))

        # Cross-attention with SAR context
        B, C, H, W = h.shape
        q  = self.cross_attn_q(h).flatten(2).permute(0, 2, 1)  # [B, H*W, C]
        k  = self.cross_attn_k(ctx)                             # [B, L, C]
        v  = self.cross_attn_v(ctx)                             # [B, L, C]
        attn = torch.softmax(q @ k.transpose(-2, -1) / (C ** 0.5), dim=-1)
        h_ctx = (attn @ v).permute(0, 2, 1).reshape(B, C, H, W)
        h = self.cross_norm(h + h_ctx)

        return h + self.skip(x)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_ldm(cfg: dict, resume_path: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg["training"].get("seed", 42))

    print("=" * 60)
    print(" SAR2EO Phase 2 — Latent Diffusion Model")
    print(f" Device: {device}")
    print("=" * 60)

    # ── VAE (frozen) ──────────────────────────────────────────────────────
    vae_enc = LatentEncoder(device)
    lat_ch  = vae_enc.latent_channels

    # ── SAR encoder ───────────────────────────────────────────────────────
    ctx_dim     = 512
    sar_encoder = SARLatentEncoder(ctx_dim=ctx_dim).to(device)

    # ── Denoising U-Net ───────────────────────────────────────────────────
    unet = LDMUNet(
        in_channels=lat_ch, out_channels=lat_ch,
        base_ch=128, ctx_dim=ctx_dim,
    ).to(device)

    n_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    n_sar    = sum(p.numel() for p in sar_encoder.parameters() if p.requires_grad)
    print(f"  LDM UNet:       {n_params/1e6:.1f}M params (trainable)")
    print(f"  SAR encoder:    {n_sar/1e6:.1f}M params (trainable)")
    print(f"  Latent size:    {lat_ch}×32×32 (vs 3×256×256 in pixel space)")
    print(f"  Compute saving: {(256*256*3) / (32*32*lat_ch):.0f}× fewer tokens\n")

    # ── Noise scheduler ────────────────────────────────────────────────────
    # DDPM registers its noise schedule as buffers, so it must be moved to the
    # device like any other module. It already uses a cosine schedule.
    scheduler = DDPM(
        timesteps=cfg.get("diffusion", {}).get("num_timesteps", 1000),
        pred_mode="eps",
    ).to(device)

    # ── Data ──────────────────────────────────────────────────────────────
    train_loader, val_loader, _ = get_dataloaders(cfg)

    # ── Optimizer ─────────────────────────────────────────────────────────
    diff_cfg = cfg.get("diffusion", {})
    lr       = diff_cfg.get("lr", 1e-4)
    optim    = torch.optim.AdamW(
        list(unet.parameters()) + list(sar_encoder.parameters()),
        lr=lr, weight_decay=1e-4,
    )
    n_epochs  = diff_cfg.get("epochs", 100)
    save_freq = cfg["training"].get("save_freq", 5)
    val_freq  = cfg["training"].get("val_freq", 10)
    ckpt_dir  = os.path.join(cfg["paths"]["checkpoint_dir"], "diffusion_ldm")
    out_dir   = os.path.join(cfg["paths"]["output_dir"], "diffusion_ldm_samples")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(out_dir,  exist_ok=True)

    scaler  = torch.amp.GradScaler(device="cuda") if device.type == "cuda" else None
    use_amp = device.type == "cuda"

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch = 1
    history     = {"train_loss": [], "val_loss": []}
    best_loss   = float("inf")

    ckpt_files  = sorted(Path(ckpt_dir).glob("epoch_*.pth")) if Path(ckpt_dir).exists() else []
    if ckpt_files:
        latest = ckpt_files[-1]
        ckpt   = torch.load(latest, map_location=device, weights_only=False)
        unet.load_state_dict(ckpt["unet"])
        sar_encoder.load_state_dict(ckpt["sar_enc"])
        optim.load_state_dict(ckpt["optim"])
        if scaler is not None and ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])
        # Without this the first validation after resuming always looks like an
        # improvement and overwrites best.pth with a worse checkpoint.
        best_loss   = ckpt.get("best_loss", float("inf"))
        history     = ckpt.get("history", history)
        start_epoch = ckpt["epoch"] + 1
        print(f"[Resume] LDM from epoch {start_epoch} | best_loss={best_loss:.5f}")
    else:
        print(f"[Train] Starting LDM from scratch")

    t_start = time.time()
    csv_path = os.path.join(cfg["paths"]["output_dir"], "losses_diffusion_ldm.csv")

    # ── Epoch loop ────────────────────────────────────────────────────────
    for epoch in range(start_epoch, n_epochs + 1):
        unet.train()
        sar_encoder.train()
        epoch_losses = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch:03d}/{n_epochs}", leave=False):
            sar    = batch["sar"].to(device)   # [B, 1, 256, 256]
            eo     = batch["eo"].to(device)     # [B, 3, 256, 256]
            B      = sar.shape[0]

            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                # Encode EO to latent space (frozen VAE)
                latents = vae_enc.encode(eo)   # [B, 4, 32, 32]

                # Sample timestep and add noise. q_sample returns (x_t, noise);
                # passing our own noise back so the loss target matches exactly.
                t     = torch.randint(0, scheduler.T, (B,), device=device)
                noise = torch.randn_like(latents)
                noisy, _ = scheduler.q_sample(latents, t, noise)

                # SAR cross-attention context
                ctx   = sar_encoder(sar)       # [B, 1024, 512]

                # Predict noise
                pred  = unet(noisy, t, ctx)

                # DDPM.loss already applies Min-SNR weighting (Hang et al. 2023),
                # so use it rather than re-deriving the weights here.
                loss  = scheduler.loss(pred, noise, t)

            optim.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                optim.step()

            epoch_losses.append(loss.item())

        mean_loss = float(np.mean(epoch_losses))
        history["train_loss"].append(mean_loss)
        elapsed = (time.time() - t_start) / 60
        eta_h   = (elapsed / epoch) * (n_epochs - epoch) / 60
        print(f"[Epoch {epoch:03d}/{n_epochs}] loss={mean_loss:.5f} | "
              f"{elapsed:.1f}min elapsed | ETA {eta_h:.1f}h")

        # Incremental CSV write
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            import csv as _csv
            w = _csv.writer(f)
            w.writerow(["epoch", "train_loss", "val_loss"])
            for i, tl in enumerate(history["train_loss"]):
                vl = history["val_loss"][i] if i < len(history["val_loss"]) else ""
                w.writerow([i + 1, tl, vl])

        # Validation: decode a few samples
        if epoch % val_freq == 0:
            unet.eval()
            sar_encoder.eval()
            val_losses = []
            sampler    = DDIMSampler(scheduler, num_steps=50)

            with torch.no_grad():
                for val_batch in val_loader:
                    v_sar  = val_batch["sar"].to(device)
                    v_eo   = val_batch["eo"].to(device)
                    B_v    = v_sar.shape[0]

                    v_lat  = vae_enc.encode(v_eo)
                    v_t    = torch.randint(0, scheduler.T, (B_v,), device=device)
                    v_noise = torch.randn_like(v_lat)
                    v_noisy, _ = scheduler.q_sample(v_lat, v_t, v_noise)
                    v_ctx   = sar_encoder(v_sar)

                    with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                        v_pred = unet(v_noisy, v_t, v_ctx)
                    val_losses.append(((v_pred - v_noise) ** 2).mean().item())

                # Generate one sample via DDIM for visual check
                v_sar_single = val_batch["sar"][:1].to(device)
                v_ctx_single = sar_encoder(v_sar_single)

                def model_fn(x, t_step):
                    t_batch = torch.full((1,), t_step, device=device, dtype=torch.long)
                    return unet(x, t_batch, v_ctx_single)

                z_sample = sampler.sample(
                    model=model_fn,
                    shape=(1, lat_ch, 32, 32),
                    device=device,
                )
                eo_sample = vae_enc.decode(z_sample)   # [1, 3, 256, 256]

                # Save triplet
                from PIL import Image as _Image
                sar_np  = val_batch["sar"][0, 0].cpu().numpy()
                sar_np  = ((sar_np + 1) / 2 * 255).clip(0, 255).astype("uint8")
                sar_rgb = np.stack([sar_np] * 3, axis=-1)
                eo_np   = ((eo_sample[0] + 1) / 2).clamp(0, 1)
                eo_np   = (eo_np.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
                grid    = np.concatenate([sar_rgb, eo_np], axis=1)
                _Image.fromarray(grid).save(
                    os.path.join(out_dir, f"epoch{epoch:03d}_sample.png")
                )

            val_loss = float(np.mean(val_losses))
            history["val_loss"].append(val_loss)
            print(f"  [Val] loss={val_loss:.5f}")

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save({
                    "epoch": epoch, "unet": unet.state_dict(),
                    "sar_enc": sar_encoder.state_dict(),
                    "val_loss": val_loss, "history": history,
                }, os.path.join(ckpt_dir, "best.pth"))
                print(f"  ✓ Best LDM checkpoint saved")

        if epoch % save_freq == 0:
            torch.save({
                "epoch": epoch, "unet": unet.state_dict(),
                "sar_enc": sar_encoder.state_dict(),
                "optim": optim.state_dict(), "history": history,
                # Required for a clean resume — see the [Resume] block above.
                "scaler": scaler.state_dict() if scaler is not None else None,
                "best_loss": best_loss,
            }, os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pth"))

    print(f"\n✓ LDM training done — {(time.time()-t_start)/60:.1f} min")
    print(f"  Best val loss: {best_loss:.5f}")
    return unet, sar_encoder


if __name__ == "__main__":
    # Windows consoles default to cp1252 and raise on the unicode used in the
    # progress output below. Force UTF-8 so local runs match Kaggle/Linux.
    import sys as _sys
    try:
        _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args   = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Inject LDM diffusion defaults
    if "diffusion" not in cfg:
        cfg["diffusion"] = {}
    cfg["diffusion"].setdefault("epochs",         100)
    cfg["diffusion"].setdefault("lr",             1e-4)
    cfg["diffusion"].setdefault("num_timesteps",  1000)

    train_ldm(cfg)
