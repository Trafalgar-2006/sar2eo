"""
train_diffusion.py — Diffusion Model Training Script

Phase 2 of the SAR2EO pipeline.
Run AFTER the GAN (train.py) has completed training.

Architecture:
  ConditionalUNet (~20M params)
    - SAR concatenated as extra channel to noisy EO
    - Sinusoidal time embedding in every ResBlock
    - Self-attention at 16×16 and 8×8
  DDPM with cosine noise schedule (T=1000)
  DDIM sampler at inference (50 steps, deterministic)

Why diffusion AFTER GAN:
  The GAN weights (best.pth) are used to pre-generate pseudo-EO images
  for fast data augmentation. The diffusion model then learns to refine
  these GAN outputs + real SAR pairs simultaneously, combining the speed
  of GAN with the perceptual quality of diffusion.

  Alternatively: train purely on real SAR/EO pairs (default mode here).

Usage:
    python train_diffusion.py --config config.yaml
    python train_diffusion.py --config config.yaml --resume checkpoints/diffusion/latest.pth

Outputs:
    checkpoints/diffusion/best.pth     — best val loss
    checkpoints/diffusion/epoch_N.pth  — periodic
    outputs/diffusion_samples/         — visual samples every val_freq epochs
"""

import os
import sys
import json
import yaml
import random
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.amp
from tqdm import tqdm
from PIL import Image

from data.dataloader       import get_dataloaders
from models.diffusion.unet import ConditionalUNet
from models.diffusion.ddpm import DDPM, DDIMSampler
from utils.ema             import EMA


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_samples(sar_batch, gen_batch, gt_batch, out_dir: str, prefix: str):
    """Save a grid of SAR / generated EO / GT EO for visual inspection."""
    os.makedirs(out_dir, exist_ok=True)
    n = min(4, len(sar_batch))
    rows = []
    for i in range(n):
        sar_np  = ((sar_batch[i][0].cpu().numpy() + 1) / 2 * 255).astype(np.uint8)
        gen_np  = ((gen_batch[i].cpu().numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        gt_np   = ((gt_batch[i].cpu().numpy()  + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        sar_rgb = np.stack([sar_np] * 3, axis=-1)
        gen_rgb = gen_np.transpose(1, 2, 0)
        gt_rgb  = gt_np.transpose(1, 2, 0)
        row     = np.concatenate([sar_rgb, gen_rgb, gt_rgb], axis=1)
        rows.append(row)
    grid = np.concatenate(rows, axis=0)
    Image.fromarray(grid).save(os.path.join(out_dir, f"{prefix}.png"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(cfg: dict, resume_path: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg["training"].get("seed", 42))

    print(f"{'='*60}")
    print(f" SAR2EO Diffusion Training")
    print(f" Device : {device}")
    print(f"{'='*60}")

    # ── Data ─────────────────────────────────────────────────────────────────
    train_loader, val_loader, _ = get_dataloaders(cfg)

    # ── Diffusion model ──────────────────────────────────────────────────────
    diff_cfg = cfg.get("diffusion", {})
    T          = diff_cfg.get("timesteps",   1000)
    base_ch    = diff_cfg.get("base_ch",       64)
    ch_mult    = tuple(diff_cfg.get("ch_mult",  [1, 2, 4, 4, 4, 4]))
    n_res      = diff_cfg.get("n_res_blocks",    2)
    attn_lvls  = tuple(diff_cfg.get("attn_levels", [4, 5]))
    dropout    = diff_cfg.get("dropout",        0.1)
    time_dim   = diff_cfg.get("time_dim",       256)
    ddim_steps = diff_cfg.get("ddim_steps",      50)

    model = ConditionalUNet(
        in_channels  = cfg["model"]["input_channels"] + cfg["model"]["output_channels"],  # 4
        out_channels = cfg["model"]["output_channels"],    # 3
        base_ch      = base_ch,
        ch_mult      = ch_mult,
        n_res_blocks = n_res,
        attn_levels  = attn_lvls,
        dropout      = dropout,
        time_dim     = time_dim,
    ).to(device)

    ddpm    = DDPM(timesteps=T, pred_mode=diff_cfg.get("pred_mode", "eps")).to(device)
    sampler = DDIMSampler(ddpm, ddim_steps=ddim_steps, eta=0.0)
    ema     = EMA(model, decay=cfg["training"].get("ema_decay", 0.9999), start_step=500)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"ConditionalUNet params : {n_params:,}")
    print(f"DDPM timesteps         : {T}")
    print(f"DDIM inference steps   : {ddim_steps}")

    # ── Optimiser ────────────────────────────────────────────────────────────
    train_cfg = cfg["training"]
    lr   = diff_cfg.get("lr", 1e-4)
    optim = torch.optim.AdamW(model.parameters(), lr=lr,
                               betas=(0.9, 0.999), weight_decay=1e-4)

    n_epochs   = diff_cfg.get("epochs", 100)
    save_freq  = train_cfg.get("save_freq",  10)
    val_freq   = train_cfg.get("val_freq",    5)
    clip_norm  = train_cfg.get("gradient_clip_norm", 1.0)
    use_amp    = train_cfg.get("mixed_precision", True) and device.type == "cuda"
    scaler     = torch.amp.GradScaler(device="cuda") if use_amp else None

    ckpt_dir   = os.path.join(cfg["paths"]["checkpoint_dir"], "diffusion")
    sample_dir = os.path.join(cfg["paths"]["output_dir"], "diffusion_samples")
    os.makedirs(ckpt_dir,   exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    # ── Cosine LR scheduler ──────────────────────────────────────────────────
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=n_epochs, eta_min=1e-6
    )

    # ── Auto-resume ──────────────────────────────────────────────────────────
    start_epoch  = 1
    best_val     = float("inf")
    global_step  = 0
    history      = {"train_loss": [], "val_loss": []}

    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if ckpt.get("ema"):
            ema.load_state_dict(ckpt["ema"])
        optim.load_state_dict(ckpt["optim"])

        # Restore the LR schedule position, then push the restored LR into the
        # optimiser — load_state_dict does not do the second part, so without it
        # the first resumed epoch trains at the scheduler's construction-time LR.
        if ckpt.get("sched"):
            sched.load_state_dict(ckpt["sched"])
            for grp, lr in zip(optim.param_groups, sched.get_last_lr()):
                grp["lr"] = lr
        if scaler is not None and ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])

        # Without this the first validation after resuming always looks like an
        # improvement and overwrites best.pth with a worse checkpoint.
        best_val    = ckpt.get("best_val", float("inf"))

        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt.get("global_step", 0)
        history     = ckpt.get("history", history)
        print(f"[Resume] from epoch {start_epoch} "
              f"| lr={optim.param_groups[0]['lr']:.2e} | best_val={best_val:.4f}")
    else:
        # Xavier init
        def init_w(m):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        model.apply(init_w)
        # Zero-init output conv for stable start
        nn.init.zeros_(model.out_conv.weight)
        nn.init.zeros_(model.out_conv.bias)

    t_start = time.time()

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, n_epochs + 1):
        model.train()
        epoch_losses = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch:03d}/{n_epochs}",
                          leave=False):
            sar     = batch["sar"].to(device)    # [B, 1, 256, 256]
            real_eo = batch["eo"].to(device)      # [B, 3, 256, 256]
            B       = sar.shape[0]
            global_step += 1

            # Sample random timesteps
            t = torch.randint(0, T, (B,), device=device)

            # Forward process: add noise to EO
            noise = torch.randn_like(real_eo)
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                x_t, noise = ddpm.q_sample(real_eo, t, noise)
                # Predict noise
                pred = model(x_t, sar, t)
                # SNR-weighted loss
                loss = ddpm.loss(pred, noise, t)

            optim.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                optim.step()

            ema.update(model)
            epoch_losses.append(loss.item())

        sched.step()
        mean_loss = float(np.mean(epoch_losses))
        history["train_loss"].append(mean_loss)
        elapsed = (time.time() - t_start) / 60

        print(
            f"[Epoch {epoch:03d}/{n_epochs}] "
            f"loss={mean_loss:.4f} "
            f"lr={optim.param_groups[0]['lr']:.2e} "
            f"| {elapsed:.1f}min"
        )

        # ── Validation ───────────────────────────────────────────────────────
        if epoch % val_freq == 0:
            model.eval()
            val_losses = []
            sar_vis, gen_vis, gt_vis = [], [], []

            with ema.apply(model):
                with torch.no_grad():
                    for vbatch in val_loader:
                        v_sar  = vbatch["sar"].to(device)
                        v_real = vbatch["eo"].to(device)
                        v_t    = torch.randint(0, T, (v_sar.shape[0],), device=device)
                        v_xt, v_noise = ddpm.q_sample(v_real, v_t)
                        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                            v_pred = model(v_xt, v_sar, v_t)
                        val_losses.append(ddpm.loss(v_pred, v_noise, v_t).item())

                    # Generate a few samples with DDIM for visualisation
                    if len(sar_vis) < 4:
                        for vbatch in val_loader:
                            v_sar  = vbatch["sar"][:4].to(device)
                            v_real = vbatch["eo"][:4]
                            gen    = sampler.sample(
                                model, v_sar,
                                shape=(min(4, v_sar.shape[0]), 3,
                                       v_sar.shape[2], v_sar.shape[3])
                            )
                            sar_vis = [v_sar[i] for i in range(v_sar.shape[0])]
                            gen_vis = [gen[i]   for i in range(gen.shape[0])]
                            gt_vis  = [v_real[i] for i in range(v_real.shape[0])]
                            break

            val_loss = float(np.mean(val_losses))
            history["val_loss"].append(val_loss)
            print(f"  [Val/EMA] loss={val_loss:.4f}")

            save_samples(sar_vis, gen_vis, gt_vis,
                         sample_dir, prefix=f"epoch{epoch:03d}")

            if val_loss < best_val:
                best_val = val_loss
                torch.save({
                    "epoch": epoch, "model": model.state_dict(),
                    "ema":   ema.state_dict(), "val_loss": val_loss,
                }, os.path.join(ckpt_dir, "best.pth"))
                print(f"  ✓ Best checkpoint saved (val={val_loss:.4f})")

        # ── Periodic save ────────────────────────────────────────────────────
        if epoch % save_freq == 0:
            torch.save({
                "epoch":       epoch,
                "global_step": global_step,
                "model":       model.state_dict(),
                "ema":         ema.state_dict(),
                "optim":       optim.state_dict(),
                # Required for a clean resume — see the [Resume] block above.
                "sched":       sched.state_dict(),
                "scaler":      scaler.state_dict() if scaler is not None else None,
                "best_val":    best_val,
                "history":     history,
            }, os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pth"))

    # Final save
    torch.save({
        "epoch": n_epochs, "model": model.state_dict(),
        "ema":   ema.state_dict(), "history": history,
    }, os.path.join(ckpt_dir, "final.pth"))
    print(f"\n✓ Diffusion training done — {(time.time()-t_start)/60:.1f} min")


# ---------------------------------------------------------------------------
# Entry point
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

    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="config.yaml")
    parser.add_argument("--resume",  default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Inject diffusion defaults if not in config
    if "diffusion" not in cfg:
        cfg["diffusion"] = {
            "timesteps":   1000,
            "base_ch":       64,
            "ch_mult":     [1, 2, 4, 4, 4, 4],
            "n_res_blocks":   2,
            "attn_levels": [4, 5],
            "dropout":      0.1,
            "time_dim":     256,
            "ddim_steps":    50,
            "pred_mode":  "eps",
            "epochs":       100,
            "lr":          1e-4,
        }

    train(cfg, resume_path=args.resume)
