"""
train.py — SAR-to-EO Image Translation Training Script

Usage:
    python train.py --config config.yaml

Supports 4 ablation configs via config.yaml `active_ablation` field:
    "l1_only"    → Config A: generator trained with L1 loss only (no GAN)
    "l1_adv"     → Config B: L1 + adversarial (standard Pix2Pix)
    "l1_adv_fft" → Config C: L1 + adversarial + FFT frequency loss
    "full"       → Config D: L1 + adversarial + FFT + VGG perceptual (MAIN MODEL)

Outputs:
    checkpoints/{ablation}/epoch_{N}.pth
    outputs/loss_curve_{ablation}.png
    outputs/losses_{ablation}.csv
    outputs/samples/{ablation}/  (sample triplets every val_freq epochs)
"""

import os
import sys
import yaml
import random
import argparse
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

from data.dataloader import get_dataloaders
from models.generator import UNetGenerator
from models.discriminator import PatchGANDiscriminator
from models.losses import GANLoss, L1Loss, FFTLoss, VGGPerceptualLoss
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
        cfg = yaml.safe_load(f)
    return cfg


def make_dirs(cfg: dict):
    ablation = cfg.get("active_ablation", "full")
    os.makedirs(os.path.join(cfg["paths"]["checkpoint_dir"], ablation), exist_ok=True)
    os.makedirs(cfg["paths"]["output_dir"], exist_ok=True)
    os.makedirs(os.path.join(cfg["paths"]["output_dir"], "samples", ablation), exist_ok=True)


def get_lr_lambda(cfg: dict):
    """Linear LR decay starting at lr_decay_start_epoch."""
    total_epochs = cfg["training"]["epochs"]
    decay_start  = cfg["training"]["lr_decay_start_epoch"]

    def lambda_rule(epoch):
        if epoch < decay_start:
            return 1.0
        progress = (epoch - decay_start) / max(1, total_epochs - decay_start)
        return max(0.0, 1.0 - progress)

    return lambda_rule


# ---------------------------------------------------------------------------
# Generator loss computation (ablation-aware)
# ---------------------------------------------------------------------------

def compute_generator_loss(
    G: UNetGenerator,
    D: PatchGANDiscriminator,
    sar: torch.Tensor,
    real_eo: torch.Tensor,
    fake_eo: torch.Tensor,
    loss_fns: dict,
    loss_weights: dict,
    ablation: str,
) -> Dict[str, torch.Tensor]:
    """
    Compute generator losses based on the active ablation config.

    Returns dict of named loss tensors (all scalar).
    """
    losses = {}

    # L1 loss — always active
    losses["G_l1"] = loss_fns["l1"](fake_eo, real_eo) * loss_weights["lambda_l1"]

    # Adversarial loss — active for l1_adv, l1_adv_fft, full
    if ablation in ("l1_adv", "l1_adv_fft", "full"):
        d_fake = D(sar, fake_eo)
        losses["G_adv"] = loss_fns["gan"](d_fake, is_real=True) * loss_weights["lambda_adv"]
    else:
        losses["G_adv"] = torch.tensor(0.0, device=sar.device)

    # FFT frequency loss — active for l1_adv_fft, full
    if ablation in ("l1_adv_fft", "full"):
        losses["G_fft"] = loss_fns["fft"](fake_eo, real_eo) * loss_weights["lambda_fft"]
    else:
        losses["G_fft"] = torch.tensor(0.0, device=sar.device)

    # VGG perceptual loss — active only for full
    if ablation == "full":
        losses["G_vgg"] = loss_fns["vgg"](fake_eo, real_eo) * loss_weights["lambda_vgg"]
    else:
        losses["G_vgg"] = torch.tensor(0.0, device=sar.device)

    # Total generator loss
    losses["G_total"] = (
        losses["G_l1"] +
        losses["G_adv"] +
        losses["G_fft"] +
        losses["G_vgg"]
    )

    return losses


# ---------------------------------------------------------------------------
# Discriminator loss computation
# ---------------------------------------------------------------------------

def compute_discriminator_loss(
    D: PatchGANDiscriminator,
    sar: torch.Tensor,
    real_eo: torch.Tensor,
    fake_eo: torch.Tensor,
    gan_loss: GANLoss,
) -> torch.Tensor:
    """Standard PatchGAN discriminator loss (real + fake, averaged)."""
    d_real = D(sar, real_eo)
    d_fake = D(sar, fake_eo.detach())   # detach — don't update G here

    loss_real = gan_loss(d_real, is_real=True)
    loss_fake = gan_loss(d_fake, is_real=False)

    return (loss_real + loss_fake) * 0.5


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg: dict):
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ablation = cfg.get("active_ablation", "full")
    use_gan  = ablation in ("l1_adv", "l1_adv_fft", "full")

    print(f"{'='*60}")
    print(f" SAR-to-EO Training")
    print(f" Ablation : {ablation}")
    print(f" Device   : {device}")
    print(f"{'='*60}")

    # ---- Reproducibility -------------------------------------------------
    set_seed(cfg["training"]["seed"])

    # ---- Data ------------------------------------------------------------
    train_loader, val_loader, _ = get_dataloaders(cfg)

    # ---- Models ----------------------------------------------------------
    model_cfg = cfg["model"]
    G = UNetGenerator(
        in_channels  = model_cfg["input_channels"],
        out_channels = model_cfg["output_channels"],
        base_ch      = model_cfg["base_channels"],
    ).to(device)
    G.init_weights()

    D = PatchGANDiscriminator(
        in_channels  = model_cfg["input_channels"],
        out_channels = model_cfg["output_channels"],
        base_ch      = model_cfg["base_channels"],
        n_layers     = model_cfg["n_layers_D"],
    ).to(device) if use_gan else None

    if D is not None:
        D.init_weights()

    n_G = sum(p.numel() for p in G.parameters() if p.requires_grad)
    n_D = sum(p.numel() for p in D.parameters() if p.requires_grad) if D else 0
    print(f"Generator params:     {n_G:,}")
    print(f"Discriminator params: {n_D:,}")

    # ---- Loss functions --------------------------------------------------
    loss_fns = {
        "l1":  L1Loss().to(device),
        "gan": GANLoss().to(device),
        "fft": FFTLoss().to(device),
        "vgg": VGGPerceptualLoss().to(device) if ablation == "full" else None,
    }
    loss_weights = cfg["loss"]

    # ---- Optimisers ------------------------------------------------------
    train_cfg = cfg["training"]
    optim_G = torch.optim.Adam(
        G.parameters(),
        lr=train_cfg["lr_generator"],
        betas=(train_cfg["beta1"], train_cfg["beta2"]),
    )
    optim_D = torch.optim.Adam(
        D.parameters(),
        lr=train_cfg["lr_discriminator"],
        betas=(train_cfg["beta1"], train_cfg["beta2"]),
    ) if D is not None else None

    # LR schedulers — linear decay from lr_decay_start_epoch
    lr_lambda   = get_lr_lambda(cfg)
    sched_G = torch.optim.lr_scheduler.LambdaLR(optim_G, lr_lambda=lr_lambda)
    sched_D = torch.optim.lr_scheduler.LambdaLR(optim_D, lr_lambda=lr_lambda) if optim_D else None

    # ---- Mixed precision -------------------------------------------------
    use_amp = train_cfg.get("mixed_precision", True) and device.type == "cuda"
    scaler_G = GradScaler() if use_amp else None
    scaler_D = GradScaler() if (use_amp and D is not None) else None

    # ---- Training state --------------------------------------------------
    n_epochs     = train_cfg["epochs"]
    save_freq    = train_cfg.get("save_freq", 10)
    val_freq     = train_cfg.get("val_freq", 5)
    ckpt_dir     = os.path.join(cfg["paths"]["checkpoint_dir"], ablation)
    output_dir   = cfg["paths"]["output_dir"]
    sample_dir   = os.path.join(output_dir, "samples", ablation)

    # Loss history (logged per epoch)
    history: Dict[str, List[float]] = {
        "G_total": [], "G_l1": [], "G_adv": [],
        "G_fft": [],   "G_vgg": [], "D_total": [],
    }

    best_val_loss = float("inf")
    t_start = time.time()

    # ---- Epoch loop ------------------------------------------------------
    for epoch in range(1, n_epochs + 1):
        G.train()
        if D is not None:
            D.train()

        epoch_losses: Dict[str, List[float]] = {k: [] for k in history}

        for batch_idx, batch in enumerate(train_loader):
            sar     = batch["sar"].to(device)      # [B, 1, 256, 256]
            real_eo = batch["eo"].to(device)       # [B, 3, 256, 256]

            # ------ Update Discriminator --------------------------------
            if D is not None and optim_D is not None:
                optim_D.zero_grad()
                with autocast(enabled=use_amp):
                    fake_eo  = G(sar).detach()
                    loss_D   = compute_discriminator_loss(D, sar, real_eo, fake_eo, loss_fns["gan"])

                if scaler_D:
                    scaler_D.scale(loss_D).backward()
                    scaler_D.step(optim_D)
                    scaler_D.update()
                else:
                    loss_D.backward()
                    optim_D.step()

                epoch_losses["D_total"].append(loss_D.item())

            # ------ Update Generator ------------------------------------
            optim_G.zero_grad()
            with autocast(enabled=use_amp):
                fake_eo  = G(sar)
                g_losses = compute_generator_loss(
                    G, D, sar, real_eo, fake_eo,
                    loss_fns, loss_weights, ablation
                )

            if scaler_G:
                scaler_G.scale(g_losses["G_total"]).backward()
                scaler_G.step(optim_G)
                scaler_G.update()
            else:
                g_losses["G_total"].backward()
                optim_G.step()

            for k in ["G_total", "G_l1", "G_adv", "G_fft", "G_vgg"]:
                epoch_losses[k].append(g_losses[k].item())

        # ---- Log epoch means -------------------------------------------
        for k in history:
            vals = epoch_losses[k]
            history[k].append(float(np.mean(vals)) if vals else 0.0)

        # ---- LR step ---------------------------------------------------
        sched_G.step()
        if sched_D:
            sched_D.step()

        elapsed = (time.time() - t_start) / 60
        print(
            f"[Epoch {epoch:03d}/{n_epochs}] "
            f"G={history['G_total'][-1]:.4f} "
            f"(l1={history['G_l1'][-1]:.3f} "
            f"adv={history['G_adv'][-1]:.3f} "
            f"fft={history['G_fft'][-1]:.3f} "
            f"vgg={history['G_vgg'][-1]:.3f}) "
            f"D={history['D_total'][-1]:.4f} "
            f"| {elapsed:.1f}min"
        )

        # ---- Validation + sample triplets ------------------------------
        if epoch % val_freq == 0:
            G.eval()
            val_losses = []
            sar_samples, pred_samples, gt_samples = [], [], []

            with torch.no_grad():
                for val_batch in val_loader:
                    v_sar    = val_batch["sar"].to(device)
                    v_real   = val_batch["eo"].to(device)
                    with autocast(enabled=use_amp):
                        v_fake = G(v_sar)
                    val_l1 = loss_fns["l1"](v_fake, v_real).item()
                    val_losses.append(val_l1)

                    if len(sar_samples) < 10:
                        sar_samples.append(v_sar[0].cpu())
                        pred_samples.append(v_fake[0].cpu())
                        gt_samples.append(v_real[0].cpu())

            val_loss = float(np.mean(val_losses))
            print(f"  [Val] L1={val_loss:.4f}")

            save_triplets(sar_samples, pred_samples, gt_samples, sample_dir,
                          prefix=f"epoch{epoch:03d}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_ckpt = os.path.join(ckpt_dir, "best.pth")
                torch.save({"epoch": epoch, "G": G.state_dict(),
                            "D": D.state_dict() if D else None,
                            "val_loss": val_loss}, best_ckpt)
                print(f"  [Val] Best checkpoint saved → {best_ckpt}")

        # ---- Periodic checkpoint save ----------------------------------
        if epoch % save_freq == 0:
            ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pth")
            torch.save({"epoch": epoch, "G": G.state_dict(),
                        "D": D.state_dict() if D else None,
                        "optim_G": optim_G.state_dict(),
                        "optim_D": optim_D.state_dict() if optim_D else None,
                        "history": history}, ckpt_path)
            print(f"  [Ckpt] Saved → {ckpt_path}")

    # ---- Final checkpoint ------------------------------------------------
    final_ckpt = os.path.join(ckpt_dir, "final.pth")
    torch.save({"epoch": n_epochs, "G": G.state_dict(),
                "D": D.state_dict() if D else None,
                "history": history}, final_ckpt)
    print(f"\n[Done] Final checkpoint → {final_ckpt}")
    print(f"[Done] Total training time: {(time.time()-t_start)/60:.1f} min")

    # ---- Save loss curves ------------------------------------------------
    plot_loss_curves(history, output_dir, ablation_name=ablation)

    return G


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAR-to-EO Training")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config YAML file")
    parser.add_argument("--ablation", type=str, default=None,
                        choices=["l1_only", "l1_adv", "l1_adv_fft", "full"],
                        help="Override active_ablation in config")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # CLI override for ablation
    if args.ablation:
        cfg["active_ablation"] = args.ablation

    make_dirs(cfg)
    train(cfg)
