"""
train_controlnet.py — Phase 3: ControlNet Fine-tuning on Stable Diffusion

Fine-tunes a ControlNet conditioned on SAR to generate EO imagery.
The frozen SD-1.5 provides the natural image prior.
The ControlNet learns SAR → EO structural conditioning.

Steps:
  1. Frozen SD-1.5 VAE encodes EO → latents
  2. DDPM adds noise to EO latents
  3. ControlNet processes SAR hint → residuals injected into SD decoder
  4. SD UNet denoises, conditioned on ControlNet residuals + neutral text
  5. MSE loss on predicted noise

Why this works so well:
  SD was trained on 2 billion natural images → it knows what realistic
  optical imagery looks like. ControlNet tells it WHERE things should be.
  Result: photorealistic EO generation guided by SAR geometry.

Run on Kaggle (single cell):
  exec(open("kaggle_train_controlnet.py", encoding="utf-8").read())

Requirements:
  pip install diffusers transformers accelerate

Expected compute: ~12-20 hrs on T4 (same session budget as GAN)
Expected quality: FID ~50-70, LPIPS ~0.18-0.22 (SOTA level)
"""

import os
import sys
import yaml
import random
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image

from data.dataloader import get_dataloaders
from models.controlnet.controlnet import SARControlNetConditioner, encode_prompt, NULL_PROMPT


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_controlnet(cfg: dict, resume_path: str = None):
    try:
        from diffusers import ControlNetModel, DDPMScheduler, AutoencoderKL, UNet2DConditionModel
        from transformers import CLIPTokenizer, CLIPTextModel
    except ImportError:
        print("ERROR: Install diffusers first:\n  pip install diffusers transformers accelerate")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg["training"].get("seed", 42))

    print("="*60)
    print(" SAR2EO ControlNet Training (Phase 3)")
    print(f" Device: {device}")
    print("="*60)

    # ── Load SD components ────────────────────────────────────────────────
    base_model = cfg.get("controlnet", {}).get("base_model", "runwayml/stable-diffusion-v1-5")
    print(f"\nLoading {base_model} ...")

    vae          = AutoencoderKL.from_pretrained(base_model, subfolder="vae").to(device)
    unet         = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet").to(device)
    tokenizer    = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder").to(device)
    scheduler    = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler")
    controlnet   = ControlNetModel.from_unet(unet).to(device)

    # Freeze everything except ControlNet
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # SAR preprocessor (1-ch → 3-ch hint)
    sar_cond = SARControlNetConditioner().to(device)

    n_trainable = (
        sum(p.numel() for p in controlnet.parameters() if p.requires_grad) +
        sum(p.numel() for p in sar_cond.parameters())
    )
    print(f"Trainable params: {n_trainable/1e6:.1f}M (ControlNet + SAR preprocessor)")
    print(f"Frozen SD params: {sum(p.numel() for p in unet.parameters())/1e6:.0f}M (frozen)")

    # Enable gradient checkpointing (saves ~50% VRAM on T4)
    unet.enable_gradient_checkpointing()
    controlnet.enable_gradient_checkpointing()

    # ── Data ──────────────────────────────────────────────────────────────
    train_loader, val_loader, _ = get_dataloaders(cfg)

    # ── Optimizer (only ControlNet + SAR preprocessor) ────────────────────
    cn_cfg = cfg.get("controlnet", {})
    lr     = cn_cfg.get("lr", 1e-5)
    optim  = torch.optim.AdamW(
        list(controlnet.parameters()) + list(sar_cond.parameters()),
        lr=lr, weight_decay=1e-2,
    )

    n_epochs  = cn_cfg.get("epochs", 30)
    save_freq = cfg["training"].get("save_freq", 5)
    ckpt_dir  = os.path.join(cfg["paths"]["checkpoint_dir"], "controlnet")
    os.makedirs(ckpt_dir, exist_ok=True)

    scaler   = torch.amp.GradScaler(device="cuda") if device.type == "cuda" else None
    use_amp  = device.type == "cuda"

    # Pre-encode the null prompt (same for all batches)
    null_emb = encode_prompt(tokenizer, text_encoder, NULL_PROMPT, device)  # [1, 77, 768]

    # ── Training loop ─────────────────────────────────────────────────────
    best_loss = float("inf")
    t_start   = time.time()
    history   = {"train_loss": []}

    for epoch in range(1, n_epochs + 1):
        controlnet.train()
        sar_cond.train()
        epoch_losses = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch:03d}/{n_epochs}", leave=False):
            sar    = batch["sar"].to(device)    # [B, 1, 256, 256]
            eo     = batch["eo"].to(device)      # [B, 3, 256, 256]
            B      = sar.shape[0]

            # Scale EO from [-1,1] to [0,1] for VAE
            eo_01  = (eo + 1) / 2

            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                # Encode EO to latent space
                with torch.no_grad():
                    latents = vae.encode(eo_01).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor  # SD scaling

                # Sample timesteps + add noise
                t  = torch.randint(0, scheduler.config.num_train_timesteps, (B,), device=device)
                noise   = torch.randn_like(latents)
                noisy_l = scheduler.add_noise(latents, noise, t)

                # SAR hint (1-ch → 3-ch)
                sar_hint = sar_cond(sar)  # [B, 3, 256, 256]

                # Null text embedding (broadcast to batch)
                enc = null_emb.expand(B, -1, -1)

                # ControlNet forward
                down_residuals, mid_residual = controlnet(
                    noisy_l, t, enc,
                    controlnet_cond=sar_hint,
                    return_dict=False,
                )

                # SD UNet forward with ControlNet residuals
                pred = unet(
                    noisy_l, t, enc,
                    down_block_additional_residuals=down_residuals,
                    mid_block_additional_residual=mid_residual,
                ).sample

                # MSE loss on predicted noise
                loss = nn.functional.mse_loss(pred, noise)

            optim.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                nn.utils.clip_grad_norm_(controlnet.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(controlnet.parameters(), 1.0)
                optim.step()

            epoch_losses.append(loss.item())

        mean_loss = float(np.mean(epoch_losses))
        history["train_loss"].append(mean_loss)
        elapsed = (time.time() - t_start) / 60
        print(f"[Epoch {epoch:03d}/{n_epochs}] loss={mean_loss:.5f} | {elapsed:.1f}min")

        if epoch % save_freq == 0:
            controlnet.save_pretrained(os.path.join(ckpt_dir, f"controlnet_epoch{epoch:03d}"))
            torch.save(sar_cond.state_dict(), os.path.join(ckpt_dir, f"sar_cond_epoch{epoch:03d}.pth"))
            print(f"  [Ckpt] Saved epoch {epoch}")

        if mean_loss < best_loss:
            best_loss = mean_loss
            controlnet.save_pretrained(os.path.join(ckpt_dir, "best_controlnet"))
            torch.save(sar_cond.state_dict(), os.path.join(ckpt_dir, "best_sar_cond.pth"))
            print(f"  ✓ Best checkpoint saved (loss={best_loss:.5f})")

    print(f"\n✓ ControlNet training done — {(time.time()-t_start)/60:.1f} min")
    print(f"  Best checkpoint: {ckpt_dir}/best_controlnet/")


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

    # Inject ControlNet defaults if missing
    if "controlnet" not in cfg:
        cfg["controlnet"] = {
            "base_model": "runwayml/stable-diffusion-v1-5",
            "epochs":     30,
            "lr":         1e-5,
        }

    train_controlnet(cfg)
