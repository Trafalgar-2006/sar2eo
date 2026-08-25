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
    with open(path, encoding="utf-8") as f:
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

    # ---- Reproducibility: capture git commit + env at start ----------------
    import subprocess as _sp
    try:
        _git_hash = _sp.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=_sp.DEVNULL
        ).decode().strip()
    except Exception:
        _git_hash = "unknown"
    _repr_meta = {
        "git_commit": _git_hash,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "cpu",
        "seed": cfg["training"].get("seed", 42),
        "ablation": ablation,
    }
    print(f"  Reproducibility: git={_git_hash} | torch={torch.__version__} | "
          f"cuda={torch.version.cuda if torch.cuda.is_available() else 'N/A'}")

    print(f"{'='*65}")
    print(f" SAR-to-EO Training — Peak Performance Build")
    print(f" Ablation : {ablation}")
    print(f" Device   : {device}")
    print(f"{'='*65}")

    set_seed(cfg["training"]["seed"])

    # ---- Data --------------------------------------------------------------
    # D7/F8: train() never uses the test split, so don't discover/split/build
    # it at all — was a third of every startup's dataset construction thrown
    # away (train.py used to unpack and discard it).
    train_loader, val_loader, _ = get_dataloaders(cfg, splits=("train", "val"))

    # ---- Models ------------------------------------------------------------
    model_cfg = cfg["model"]
    G = UNetGenerator(
        in_channels   = model_cfg["input_channels"],
        out_channels  = model_cfg["output_channels"],
        base_ch       = model_cfg.get("base_ch", 64),
        use_attention = model_cfg.get("use_attention", True),
        pretrained    = model_cfg.get("pretrained_encoder", True),
        gradient_checkpointing = model_cfg.get("gradient_checkpointing", False),
        full_res_skip = model_cfg.get("full_res_skip", True),
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
    # Optional cap on epochs per invocation, for 12h-limited Kaggle sessions.
    # None = run straight through to n_epochs.
    session_limit = train_cfg.get("session_epoch_limit") or None
    # Wall-clock cap, for a fixed lab slot ("the GPU is mine 4 hours a day").
    # Checked after each epoch, so a run always stops on an epoch boundary with
    # a complete checkpoint — never part-way through one.
    session_minutes = train_cfg.get("session_time_limit_minutes") or None
    ckpt_dir  = os.path.join(cfg["paths"]["checkpoint_dir"], ablation)
    out_dir   = cfg["paths"]["output_dir"]
    log_dir   = cfg["paths"]["log_dir"]
    sample_dir= os.path.join(out_dir, "samples", ablation)
    step_log  = os.path.join(log_dir, f"{ablation}_steps.jsonl")

    # Validation figures are recorded per-epoch alongside the training losses so
    # the train/val gap is visible in the CSV and the plot. Without them the only
    # record of validation is a printed line that scrolls past, and overfitting —
    # train loss falling while val loss rises — cannot be seen at all.
    #
    # `train_l1` is the UNWEIGHTED L1, which `val_l1` can be compared against
    # directly; `G_l1` is multiplied by lambda_l1 (100) and is not comparable.
    VAL_KEYS = ("val_l1", "val_lpips")
    history: Dict[str, List[float]] = {
        "G_total": [], "G_l1": [], "G_adv": [],
        "G_fft": [],  "G_vgg": [], "G_ssim": [], "D_total": [],
        "train_l1": [], "val_l1": [], "val_lpips": [],
    }

    best_val_loss = float("inf")
    # OQ-5: perceptual model-selection, tracked alongside (never instead
    # of) the L1 criterion that picks best.pth.
    best_val_lpips = float("inf")
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
            # Checkpoints written before validation tracking existed have no
            # val columns; backfill so the CSV and plot stay rectangular.
            n_done = len(history["G_total"])
            for _k, _fill in (("train_l1", 0.0), ("val_l1", float("nan")),
                              ("val_lpips", float("nan"))):
                if _k not in history:
                    history[_k] = [_fill] * n_done

        # Restore LR schedule position. Without this the scheduler restarts at
        # step 0, so a resumed session re-runs warmup and cosine from the top —
        # spiking the encoder LR back to full mid-training.
        if ckpt.get("sched_G"):
            sched_G.load_state_dict(ckpt["sched_G"])
        if sched_D and ckpt.get("sched_D"):
            sched_D.load_state_dict(ckpt["sched_D"])

        # load_state_dict restores the schedule's position but does NOT write the
        # LR back into the optimiser — param_groups still hold the values set
        # when the scheduler was constructed (warmup start, ~1% of base). Push
        # the restored LR through, or the first resumed epoch trains at ~2e-7.
        for grp, lr in zip(optim_G.param_groups, sched_G.get_last_lr()):
            grp["lr"] = lr
        if sched_D is not None and optim_D is not None:
            for grp, lr in zip(optim_D.param_groups, sched_D.get_last_lr()):
                grp["lr"] = lr
        if scaler_G and ckpt.get("scaler_G"):
            scaler_G.load_state_dict(ckpt["scaler_G"])
        if scaler_D and ckpt.get("scaler_D"):
            scaler_D.load_state_dict(ckpt["scaler_D"])

        # Restore the best-so-far val loss, otherwise the first validation of a
        # resumed session overwrites best.pth with a worse checkpoint.
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        best_val_lpips = ckpt.get("best_val_lpips", float("inf"))

        global_step = ckpt.get("global_step", 0)
        start_epoch = ckpt["epoch"] + 1
        print(f"[Resume] Resuming from epoch {start_epoch}/{n_epochs} "
              f"| lr_enc={optim_G.param_groups[0]['lr']:.2e} "
              f"| best_val={best_val_loss:.4f}")
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

        # Validation keys are filled by the validation block, not by per-batch
        # accumulation, so they are excluded here.
        epoch_losses: Dict[str, List[float]] = {
            k: [] for k in history if k not in VAL_KEYS
        }

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
            # Unweighted, so it can be compared directly against val_l1.
            epoch_losses["train_l1"].append(
                g_losses["G_l1"].item() / max(loss_weights["lambda_l1"], 1e-8))

            # Per-step logging (every 50 steps)
            if global_step % 50 == 0:
                log_entry = {
                    "step": global_step, "epoch": epoch,
                    **{k: g_losses[k].item() for k in ["G_total", "G_l1", "G_adv",
                                                        "G_fft", "G_vgg", "G_ssim"]},
                    "D_total": loss_D.item() if D else 0.0,
                }
                with open(step_log, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry) + "\n")

        # ---- Log epoch means -----------------------------------------------
        for k in history:
            if k in VAL_KEYS:
                # Placeholder for every epoch; the validation block overwrites
                # the last entry on the epochs where it actually runs. NaN so
                # plots draw a gap rather than a misleading zero.
                history[k].append(float("nan"))
                continue
            vals = epoch_losses[k]
            history[k].append(float(np.mean(vals)) if vals else 0.0)

        # ---- LR step -------------------------------------------------------
        sched_G.step()
        if sched_D:
            sched_D.step()

        elapsed = (time.time() - t_start) / 60
        enc_lr  = optim_G.param_groups[0]["lr"]
        dec_lr  = optim_G.param_groups[1]["lr"]
        eta_h   = (elapsed / epoch) * (n_epochs - epoch) / 60  # hours remaining
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
            f"| {elapsed:.1f}min elapsed | ETA {eta_h:.1f}h"
        )

        # ---- INCREMENTAL CSV WRITE (every epoch — survives Kaggle timeout) --
        # This ensures loss history is never lost even if training is interrupted.
        import csv
        csv_path = os.path.join(out_dir, f"losses_{ablation}.csv")
        os.makedirs(out_dir, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as csvf:
            writer = csv.writer(csvf)
            writer.writerow(["epoch"] + list(history.keys()))
            for i in range(len(history["G_total"])):
                row = [i + 1] + [history[k][i] for k in history]
                writer.writerow(row)

        # ---- Validation (using EMA model) ----------------------------------
        if epoch % val_freq == 0:
            G.eval()
            val_losses = []
            val_lpips_vals: List[float] = []
            # Lazily imported: a missing `lpips` package degrades to
            # L1-only model-selection instead of killing a live run.
            try:
                from utils.metrics import compute_lpips
                _lpips_ok = True
            except Exception as _e:      # pragma: no cover - env dependent
                print(f"  [Val] lpips unavailable ({_e}) - best_lpips.pth disabled")
                _lpips_ok = False
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

                        if _lpips_ok:
                            try:
                                val_lpips_vals.append(compute_lpips(
                                    v_fake.float(), v_real.float(), device=device))
                            except Exception as _e:
                                print(f"  [Val] LPIPS failed ({_e}) - "
                                      f"perceptual selection disabled")
                                _lpips_ok = False
                                val_lpips_vals.clear()

                        if len(sar_samples) < 10:
                            sar_samples.append(v_sar[0].cpu())
                            pred_samples.append(v_fake[0].cpu())
                            gt_samples.append(v_real[0].cpu())

            val_loss = float(np.mean(val_losses))
            # Overwrite this epoch's NaN placeholder so the gap is recorded.
            history["val_l1"][-1] = val_loss

            gap = val_loss - history["train_l1"][-1]
            print(f"  [Val/EMA] L1={val_loss:.4f}  "
                  f"(train={history['train_l1'][-1]:.4f}, gap={gap:+.4f})")
            if gap > 0.05 and epoch > train_cfg.get("warmup_epochs", 5) * 3:
                print(f"  [Val] NOTE: val L1 exceeds train L1 by {gap:.3f}. "
                      f"A gap that widens over epochs means overfitting.")

            save_triplets(sar_samples, pred_samples, gt_samples,
                          sample_dir, prefix=f"epoch{epoch:03d}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(ckpt_dir, "best.pth")
                torch.save({
                    "epoch":    epoch,
                    "G":        G.state_dict(),
                    "G_ema":    ema.state_dict(),
                    "D":        D.state_dict() if D else None,
                    "val_loss": val_loss,
                    "meta":     _repr_meta,   # reproducibility
                }, best_path)
                print(f"  [Val] ✓ Best checkpoint saved → {best_path}")

            # OQ-5 (additive): a second checkpoint selected on perception,
            # not pixels. best.pth above is untouched - anything already
            # pointing at it keeps the exact same behaviour.
            if val_lpips_vals:
                val_lpips = float(np.mean(val_lpips_vals))
                history["val_lpips"][-1] = val_lpips
                print(f"  [Val/EMA] LPIPS={val_lpips:.4f}")
                if val_lpips < best_val_lpips:
                    best_val_lpips = val_lpips
                    lpips_path = os.path.join(ckpt_dir, "best_lpips.pth")
                    torch.save({
                        "epoch":      epoch,
                        "G":          G.state_dict(),
                        "G_ema":      ema.state_dict(),
                        "D":          D.state_dict() if D else None,
                        "val_loss":   val_loss,
                        "val_lpips":  val_lpips,
                        "meta":       _repr_meta,
                    }, lpips_path)
                    print(f"  [Val] ✓ Best LPIPS checkpoint saved → {lpips_path}")

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
                # Scheduler/scaler state and best_val_loss are required for a
                # clean multi-session resume — see the [Resume] block above.
                "sched_G":     sched_G.state_dict(),
                "sched_D":     sched_D.state_dict() if sched_D else None,
                "scaler_G":    scaler_G.state_dict() if scaler_G else None,
                "scaler_D":    scaler_D.state_dict() if scaler_D else None,
                "best_val_loss": best_val_loss,
                "best_val_lpips": best_val_lpips,
                "history":     history,
                "meta":        _repr_meta,   # reproducibility
            }, ckpt_path)
            print(f"  [Ckpt] Saved → {ckpt_path}")

        # ---- Stop early if this session has run its quota -------------------
        # Kaggle kills a session at 12h, so long runs are split across sessions.
        # `epochs` stays at the true total (so the cosine schedule spans the
        # whole run and does not jump when a later session resumes); this only
        # caps how many epochs any single session executes.
        epochs_this_session = epoch - start_epoch + 1
        mins_elapsed        = (time.time() - t_start) / 60
        mean_epoch          = mins_elapsed / max(1, epochs_this_session)
        hit_epoch_cap = bool(session_limit) and epochs_this_session >= session_limit
        # Stop when the NEXT epoch would overrun the slot, judged by this
        # session's own mean epoch time. Stopping only once already over would
        # overshoot the slot by a full epoch every single time.
        hit_time_cap = (session_minutes is not None
                        and mins_elapsed + mean_epoch > session_minutes)
        if (hit_epoch_cap or hit_time_cap) and epoch < n_epochs:
            if epoch % save_freq != 0:   # make sure this session's work survives
                torch.save({
                    "epoch": epoch, "global_step": global_step,
                    "G": G.state_dict(), "G_ema": ema.state_dict(),
                    "D": D.state_dict() if D else None,
                    "optim_G": optim_G.state_dict(),
                    "optim_D": optim_D.state_dict() if optim_D else None,
                    "sched_G": sched_G.state_dict(),
                    "sched_D": sched_D.state_dict() if sched_D else None,
                    "scaler_G": scaler_G.state_dict() if scaler_G else None,
                    "scaler_D": scaler_D.state_dict() if scaler_D else None,
                    "best_val_loss": best_val_loss,
                    "best_val_lpips": best_val_lpips,
                    "history": history, "meta": _repr_meta,
                }, os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pth"))
            why       = "time budget" if hit_time_cap else "epoch budget"
            remaining = n_epochs - epoch
            print(f"\n[Session] Stopping on {why}: {epochs_this_session} epoch(s) "
                  f"in {mins_elapsed:.1f} min, at epoch {epoch}/{n_epochs}.")
            print(f"[Session] {remaining} epoch(s) left, "
                  f"~{remaining * mean_epoch / 60:.1f} h at {mean_epoch:.1f} min/epoch.")
            print(f"[Session] Rerun the same command to resume at epoch {epoch + 1} — "
                  f"optimiser, LR schedule and best-so-far are all restored.")
            return G

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
    # Windows consoles default to cp1252 and raise on the unicode used in the
    # progress output below. Force UTF-8 so local runs match Kaggle/Linux.
    import sys as _sys
    try:
        _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

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
