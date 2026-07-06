"""
train.py — SAR-to-EO Training Script (Peak Performance Build)

Architecture: ResNet50-UNet + CBAM attention generator
              Multi-scale (3×) PatchGAN discriminator

Key training improvements over baseline:
  - Differential learning rates: encoder (pretrained) 10× lower than decoder
  - Cosine annealing with linear warmup (5 epochs) — safe for pretrained encoder
  - Gradient clipping (max_norm=1.0) — prevents GAN training collapse
  - EMA generator weights — used for validation and saved as final model
  - MS-SSIM loss added to the loss stack
  - Per-step loss logging to JSONL for smooth curve plotting

Usage:
    python train.py --config config.yaml
    python train.py --config config.yaml --ablation full

Ablation configs:
    "l1_only"    → Config A: L1 only (no GAN)
    "l1_adv"     → Config B: L1 + multi-scale adversarial
    "l1_adv_fft" → Config C: + FFT frequency loss
    "full"       → Config D: + VGG + MS-SSIM (MAIN MODEL)

Outputs:
    checkpoints/{ablation}/best.pth      — EMA weights, best val loss
    checkpoints/{ablation}/epoch_N.pth   — periodic saves
    checkpoints/{ablation}/final.pth     — last epoch
    outputs/loss_curve_{ablation}.png
    outputs/losses_{ablation}.csv
    logs/{ablation}_steps.jsonl          — per-step loss log
"""

import os
import sys
import json
import yaml
import random
import argparse
import time
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.amp
from torch.optim.lr_scheduler import (
    LinearLR, CosineAnnealingLR, SequentialLR
)

from data.dataloader import get_dataloaders
from models.generator import UNetGenerator
from models.discriminator import MultiScaleDiscriminator
from models.losses import GANLoss, L1Loss, FFTLoss, VGGPerceptualLoss, MSSSIMLoss
from utils.ema import EMA
from utils.visualize import plot_loss_curves, save_triplets


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_dirs(cfg: dict):
    ablation = cfg.get("active_ablation", "full")
    for d in [
        os.path.join(cfg["paths"]["checkpoint_dir"], ablation),
        cfg["paths"]["output_dir"],
        os.path.join(cfg["paths"]["output_dir"], "samples", ablation),
        cfg["paths"]["log_dir"],
    ]:
        os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# Learning rate schedulers (cosine warmup)
# ---------------------------------------------------------------------------

def build_schedulers(cfg: dict, optim_G, optim_D):
    """
    Build cosine annealing with linear warmup for G and D.

    Warmup: LR linearly increases from lr_min to lr for warmup_epochs.
    Cosine: LR then decays from lr to lr_min over remaining epochs.

    This is critical for the pretrained ResNet50 encoder — large initial LR
    updates would destroy the pretrained features.
    """
    train_cfg     = cfg["training"]
    n_epochs      = train_cfg["epochs"]
    warmup_epochs = train_cfg.get("warmup_epochs", 5)
    lr_min        = train_cfg.get("lr_min", 1e-6)
    cosine_epochs = max(1, n_epochs - warmup_epochs)

    def make_sched(optim):
        warmup = LinearLR(optim, start_factor=0.01, end_factor=1.0,
                          total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(optim, T_max=cosine_epochs, eta_min=lr_min)
        return SequentialLR(optim, schedulers=[warmup, cosine],
                            milestones=[warmup_epochs])

    sched_G = make_sched(optim_G)
    sched_D = make_sched(optim_D) if optim_D is not None else None
    return sched_G, sched_D


# ---------------------------------------------------------------------------
# Generator loss computation (ablation-aware)
# ---------------------------------------------------------------------------

def compute_generator_loss(
    D: MultiScaleDiscriminator,
    sar: torch.Tensor,
    real_eo: torch.Tensor,
    fake_eo: torch.Tensor,
    loss_fns: dict,
    loss_weights: dict,
    ablation: str,
) -> Dict[str, torch.Tensor]:
    """
    Compute generator losses based on the active ablation config.
    Returns a dict of named scalar loss tensors.
    """
    losses = {}

    # L1 — always active
    losses["G_l1"] = loss_fns["l1"](fake_eo, real_eo) * loss_weights["lambda_l1"]

    # Adversarial (multi-scale) — l1_adv, l1_adv_fft, full
    if ablation in ("l1_adv", "l1_adv_fft", "full"):
        d_fakes = D(sar, fake_eo)    # list of patch maps
        losses["G_adv"] = loss_fns["gan"](d_fakes, is_real=True) * loss_weights["lambda_adv"]
    else:
        losses["G_adv"] = torch.tensor(0.0, device=sar.device)

    # FFT — l1_adv_fft, full
    if ablation in ("l1_adv_fft", "full"):
        losses["G_fft"] = loss_fns["fft"](fake_eo, real_eo) * loss_weights["lambda_fft"]
    else:
        losses["G_fft"] = torch.tensor(0.0, device=sar.device)

    # VGG + MS-SSIM — full only
    if ablation == "full":
        losses["G_vgg"]  = loss_fns["vgg"](fake_eo, real_eo)  * loss_weights["lambda_vgg"]
        losses["G_ssim"] = loss_fns["ssim"](fake_eo, real_eo) * loss_weights["lambda_ssim"]
    else:
        losses["G_vgg"]  = torch.tensor(0.0, device=sar.device)
        losses["G_ssim"] = torch.tensor(0.0, device=sar.device)

    losses["G_total"] = (
        losses["G_l1"]  + losses["G_adv"] + losses["G_fft"] +
        losses["G_vgg"] + losses["G_ssim"]
    )
    return losses


# ---------------------------------------------------------------------------
# Discriminator loss computation
# ---------------------------------------------------------------------------

def compute_discriminator_loss(
    D: MultiScaleDiscriminator,
    sar: torch.Tensor,
    real_eo: torch.Tensor,
    fake_eo: torch.Tensor,
    gan_loss: GANLoss,
) -> torch.Tensor:
    """Multi-scale PatchGAN discriminator loss."""
    d_reals = D(sar, real_eo)
    d_fakes = D(sar, fake_eo.detach())
    loss_real = gan_loss(d_reals, is_real=True)
    loss_fake = gan_loss(d_fakes, is_real=False)
    return (loss_real + loss_fake) * 0.5


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg: dict):
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ablation = cfg.get("active_ablation", "full")
    use_gan  = ablation in ("l1_adv", "l1_adv_fft", "full")

    print(f"{'='*65}")
    print(f" SAR-to-EO Training — Peak Performance Build")
    print(f" Ablation : {ablation}")
    print(f" Device   : {device}")
    print(f"{'='*65}")

    set_seed(cfg["training"]["seed"])

    # ---- Data --------------------------------------------------------------
    train_loader, val_loader, _ = get_dataloaders(cfg)

    # ---- Models ------------------------------------------------------------
    model_cfg = cfg["model"]
    G = UNetGenerator(
        in_channels   = model_cfg["input_channels"],
        out_channels  = model_cfg["output_channels"],
        base_ch       = model_cfg.get("base_ch", 64),
        use_attention = model_cfg.get("use_attention", True),
        pretrained    = model_cfg.get("pretrained_encoder", True),
        gradient_checkpointing = model_cfg.get("gradient_checkpointing", False),
    ).to(device)
    G.init_weights()   # initialises decoder/projections; encoder stays pretrained

    D = MultiScaleDiscriminator(
        in_channels  = model_cfg["input_channels"],
        out_channels = model_cfg["output_channels"],
        base_ch      = 64,
        n_layers     = model_cfg.get("n_layers_D", 3),
        n_scales     = model_cfg.get("n_scales_D", 3),
    ).to(device) if use_gan else None

    if D is not None:
        D.init_weights()

    # EMA for G — shadow model used for validation and saved as best weights
    ema = EMA(G, decay=cfg["training"].get("ema_decay", 0.999), start_step=100)

    n_G = sum(p.numel() for p in G.parameters())
    n_D = sum(p.numel() for p in D.parameters()) if D else 0
    print(f"Generator params     : {n_G:,}  "
          f"(encoder={sum(p.numel() for p in G.encoder_parameters()):,} pretrained)")
    print(f"Discriminator params : {n_D:,}")

    # ---- Loss functions (all always initialised — avoids ablation-switch crashes)
    loss_fns = {
        "l1":   L1Loss().to(device),
        "gan":  GANLoss().to(device),
        "fft":  FFTLoss().to(device),
        "vgg":  VGGPerceptualLoss().to(device),
        "ssim": MSSSIMLoss().to(device),
    }
    loss_weights = cfg["loss"]
    clip_norm    = cfg["training"].get("gradient_clip_norm", 1.0)

    # ---- Optimisers (differential LR for G) --------------------------------
    train_cfg = cfg["training"]
    lr_enc = train_cfg.get("lr_encoder", 2e-5)
    lr_dec = train_cfg.get("lr_decoder", 2e-4)
    lr_D   = train_cfg.get("lr_discriminator", 2e-4)
    b1, b2 = train_cfg["beta1"], train_cfg["beta2"]

    optim_G = torch.optim.Adam([
        {"params": G.encoder_parameters(), "lr": lr_enc},   # pretrained — gentle
        {"params": G.decoder_parameters(), "lr": lr_dec},   # random init — normal
    ], betas=(b1, b2))

    optim_D = torch.optim.Adam(
        D.parameters(), lr=lr_D, betas=(b1, b2)
    ) if D is not None else None

    # ---- Schedulers (cosine warmup) ----------------------------------------
    sched_G, sched_D = build_schedulers(cfg, optim_G, optim_D)

    # ---- Mixed precision ----------------------------------------------------
    use_amp   = train_cfg.get("mixed_precision", True) and device.type == "cuda"
    scaler_G  = torch.amp.GradScaler(device="cuda") if use_amp else None
    scaler_D  = torch.amp.GradScaler(device="cuda") if (use_amp and D) else None

    # ---- Training state -----------------------------------------------------
    n_epochs  = train_cfg["epochs"]
    save_freq = train_cfg.get("save_freq", 10)
    val_freq  = train_cfg.get("val_freq", 5)
    ckpt_dir  = os.path.join(cfg["paths"]["checkpoint_dir"], ablation)
    out_dir   = cfg["paths"]["output_dir"]
    log_dir   = cfg["paths"]["log_dir"]
    sample_dir= os.path.join(out_dir, "samples", ablation)
    step_log  = os.path.join(log_dir, f"{ablation}_steps.jsonl")

    history: Dict[str, List[float]] = {
        "G_total": [], "G_l1": [], "G_adv": [],
        "G_fft": [],  "G_vgg": [], "G_ssim": [], "D_total": [],
    }

    best_val_loss = float("inf")
    start_epoch   = 1
    global_step   = 0

    # ---- Auto-resume from latest checkpoint ---------------------------------
    ckpt_files = sorted(Path(ckpt_dir).glob("epoch_*.pth")) if Path(ckpt_dir).exists() else []
    if ckpt_files:
        latest = ckpt_files[-1]
        print(f"\n[Resume] Found checkpoint: {latest}")
        ckpt = torch.load(latest, map_location=device, weights_only=False)
        G.load_state_dict(ckpt["G"])
        if D and ckpt.get("D"):
            D.load_state_dict(ckpt["D"])
        if ckpt.get("G_ema"):
            ema.load_state_dict(ckpt["G_ema"])
        if ckpt.get("optim_G"):
            optim_G.load_state_dict(ckpt["optim_G"])
        if D and ckpt.get("optim_D"):
            optim_D.load_state_dict(ckpt["optim_D"])
        if ckpt.get("history"):
            history = ckpt["history"]
            # Ensure G_ssim key exists (backward compat)
            if "G_ssim" not in history:
                history["G_ssim"] = [0.0] * len(history["G_total"])
        global_step = ckpt.get("global_step", 0)
        start_epoch = ckpt["epoch"] + 1
        print(f"[Resume] Resuming from epoch {start_epoch}/{n_epochs}")
    else:
        print(f"[Train] Starting from scratch")

    if start_epoch > n_epochs:
        print(f"[Done] Training already complete ({n_epochs} epochs).")
        return G

    t_start = time.time()

    # ---- Epoch loop ---------------------------------------------------------
    for epoch in range(start_epoch, n_epochs + 1):
        G.train()
        if D is not None:
            D.train()

        epoch_losses: Dict[str, List[float]] = {k: [] for k in history}

        for batch_idx, batch in enumerate(train_loader):
            sar     = batch["sar"].to(device)       # [B, 1, 256, 256]
            real_eo = batch["eo"].to(device)         # [B, 3, 256, 256]
            global_step += 1

            # ------ Update Discriminator ------------------------------------
            if D is not None and optim_D is not None:
                optim_D.zero_grad()
                with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                    fake_eo = G(sar).detach()
                    loss_D  = compute_discriminator_loss(
                        D, sar, real_eo, fake_eo, loss_fns["gan"]
                    )
                if scaler_D:
                    scaler_D.scale(loss_D).backward()
                    scaler_D.unscale_(optim_D)
                    nn.utils.clip_grad_norm_(D.parameters(), clip_norm)
                    scaler_D.step(optim_D)
                    scaler_D.update()
                else:
                    loss_D.backward()
                    nn.utils.clip_grad_norm_(D.parameters(), clip_norm)
                    optim_D.step()
                epoch_losses["D_total"].append(loss_D.item())

            # ------ Update Generator ----------------------------------------
            optim_G.zero_grad()
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                fake_eo  = G(sar)
                g_losses = compute_generator_loss(
                    D, sar, real_eo, fake_eo, loss_fns, loss_weights, ablation
                )
            if scaler_G:
                scaler_G.scale(g_losses["G_total"]).backward()
                scaler_G.unscale_(optim_G)
                nn.utils.clip_grad_norm_(G.parameters(), clip_norm)
                scaler_G.step(optim_G)
                scaler_G.update()
            else:
                g_losses["G_total"].backward()
                nn.utils.clip_grad_norm_(G.parameters(), clip_norm)
                optim_G.step()

            # Update EMA after each G step
            ema.update(G)

            for k in ["G_total", "G_l1", "G_adv", "G_fft", "G_vgg", "G_ssim"]:
                epoch_losses[k].append(g_losses[k].item())

            # Per-step logging (every 50 steps)
            if global_step % 50 == 0:
                log_entry = {
                    "step": global_step, "epoch": epoch,
                    **{k: g_losses[k].item() for k in ["G_total", "G_l1", "G_adv",
                                                        "G_fft", "G_vgg", "G_ssim"]},
                    "D_total": loss_D.item() if D else 0.0,
                }
                with open(step_log, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")

        # ---- Log epoch means -----------------------------------------------
        for k in history:
            vals = epoch_losses[k]
            history[k].append(float(np.mean(vals)) if vals else 0.0)

        # ---- LR step -------------------------------------------------------
        sched_G.step()
        if sched_D:
            sched_D.step()

        elapsed = (time.time() - t_start) / 60
        enc_lr  = optim_G.param_groups[0]["lr"]
        dec_lr  = optim_G.param_groups[1]["lr"]
        print(
            f"[Epoch {epoch:03d}/{n_epochs}] "
            f"G={history['G_total'][-1]:.4f} "
            f"(l1={history['G_l1'][-1]:.3f} "
            f"adv={history['G_adv'][-1]:.3f} "
            f"fft={history['G_fft'][-1]:.3f} "
            f"vgg={history['G_vgg'][-1]:.3f} "
            f"ssim={history['G_ssim'][-1]:.3f}) "
            f"D={history['D_total'][-1]:.4f} "
            f"| lr_enc={enc_lr:.2e} lr_dec={dec_lr:.2e} "
            f"| {elapsed:.1f}min"
        )

        # ---- Validation (using EMA model) ----------------------------------
        if epoch % val_freq == 0:
            G.eval()
            val_losses = []
            sar_samples, pred_samples, gt_samples = [], [], []

            with ema.apply(G):     # temporarily load EMA weights
                with torch.no_grad():
                    for val_batch in val_loader:
                        v_sar  = val_batch["sar"].to(device)
                        v_real = val_batch["eo"].to(device)
                        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                            v_fake = G(v_sar)
                        val_l1 = loss_fns["l1"](v_fake, v_real).item()
                        val_losses.append(val_l1)

                        if len(sar_samples) < 10:
                            sar_samples.append(v_sar[0].cpu())
                            pred_samples.append(v_fake[0].cpu())
                            gt_samples.append(v_real[0].cpu())

            val_loss = float(np.mean(val_losses))
            print(f"  [Val/EMA] L1={val_loss:.4f}")

            save_triplets(sar_samples, pred_samples, gt_samples,
                          sample_dir, prefix=f"epoch{epoch:03d}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(ckpt_dir, "best.pth")
                torch.save({
                    "epoch":    epoch,
                    "G":        G.state_dict(),
                    "G_ema":    ema.state_dict(),   # EMA weights saved here
                    "D":        D.state_dict() if D else None,
                    "val_loss": val_loss,
                }, best_path)
                print(f"  [Val] ✓ Best checkpoint saved → {best_path}")

        # ---- Periodic checkpoint -------------------------------------------
        if epoch % save_freq == 0:
            ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pth")
            torch.save({
                "epoch":       epoch,
                "global_step": global_step,
                "G":           G.state_dict(),
                "G_ema":       ema.state_dict(),
                "D":           D.state_dict() if D else None,
                "optim_G":     optim_G.state_dict(),
                "optim_D":     optim_D.state_dict() if optim_D else None,
                "history":     history,
            }, ckpt_path)
            print(f"  [Ckpt] Saved → {ckpt_path}")

    # ---- Final checkpoint ---------------------------------------------------
    final_path = os.path.join(ckpt_dir, "final.pth")
    torch.save({
        "epoch":   n_epochs,
        "G":       G.state_dict(),
        "G_ema":   ema.state_dict(),
        "D":       D.state_dict() if D else None,
        "history": history,
    }, final_path)
    print(f"\n[Done] Final checkpoint → {final_path}")
    print(f"[Done] Total training time: {(time.time()-t_start)/60:.1f} min")

    # ---- Save loss curves ---------------------------------------------------
    plot_loss_curves(history, out_dir, ablation_name=ablation)

    return G


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAR-to-EO Training")
    parser.add_argument("--config",   type=str, default="config.yaml")
    parser.add_argument("--ablation", type=str, default=None,
                        choices=["l1_only", "l1_adv", "l1_adv_fft", "full"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.ablation:
        cfg["active_ablation"] = args.ablation

    make_dirs(cfg)
    train(cfg)
