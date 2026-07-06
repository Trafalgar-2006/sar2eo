"""
SINGLE CELL — SAR2EO Phase 2: Latent Diffusion Model on Kaggle

Paste this ENTIRE block into one Kaggle cell and run.
Run AFTER Phase 1 GAN is complete and you have best.pth saved.

Session budget: ~8-10hrs on T4 (100 epochs LDM in latent space)
"""

import subprocess, os, sys, shutil

# ── 1. Clone / pull repo ────────────────────────────────────────────────────
REPO = "/kaggle/working/sar2eo"
if os.path.exists(REPO):
    subprocess.run(f"cd {REPO} && git pull --quiet", shell=True)
else:
    subprocess.run(
        f"git clone --quiet https://github.com/Trafalgar-2006/sar2eo.git {REPO}",
        shell=True, check=True
    )
sys.path.insert(0, REPO)
os.chdir(REPO)
print("✓ Repo ready")

# ── 2. Install deps ──────────────────────────────────────────────────────────
subprocess.run("pip install -q lpips diffusers transformers accelerate", shell=True, check=True)
print("✓ Deps installed (includes diffusers for SD-1.5 VAE)")

# ── 3. Restore Phase 1 GAN best.pth if provided as input ────────────────────
# Add your Phase 1 Kaggle output as an Input Dataset before running
GAN_CKPT = "/kaggle/working/checkpoints/full"
os.makedirs(GAN_CKPT, exist_ok=True)

for dirpath, _, filenames in os.walk("/kaggle/input"):
    for f in filenames:
        if f == "best.pth" and "full" in dirpath:
            src = os.path.join(dirpath, f)
            dst = os.path.join(GAN_CKPT, "best.pth")
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
                print(f"✓ Restored GAN best.pth from: {dirpath}")

# ── 4. Auto-discover dataset ─────────────────────────────────────────────────
INPUT_ROOT  = "/kaggle/input"
KAGGLE_DATA = None
TERRAIN_KEYS = {"agri", "urban", "grassland", "barrenland"}

for dirpath, dirnames, _ in os.walk(INPUT_ROOT):
    if len({d.lower() for d in dirnames} & TERRAIN_KEYS) >= 2:
        KAGGLE_DATA = dirpath
        break

if KAGGLE_DATA is None:
    raise RuntimeError("❌ Dataset not found — add Sentinel-1&2 dataset as Input")
print(f"✓ Dataset: {KAGGLE_DATA}")

# ── 5. Write config ──────────────────────────────────────────────────────────
import yaml

cfg = {
    "active_ablation": "full",
    "paths": {
        "dataset_dir":    KAGGLE_DATA,
        "checkpoint_dir": "/kaggle/working/checkpoints",
        "output_dir":     "/kaggle/working/outputs",
        "log_dir":        "/kaggle/working/logs",
    },
    "data": {
        "dataset_type":  "kaggle",
        "split_strategy":"random",
        "image_size":    256,
        "num_workers":   2,
        "augment_train": True,
    },
    "model": {
        "input_channels":  1,
        "output_channels": 3,
        "base_ch":         64,
        "use_attention":   True,
    },
    "training": {
        "seed":       42,
        "save_freq":  5,
        "val_freq":   10,
        "batch_size": 8,
    },
    "diffusion": {
        "epochs":        100,    # ~8-10hrs on T4
        "lr":            1e-4,
        "num_timesteps": 1000,
    },
}

CFG_PATH = "/kaggle/working/config_diffusion.yaml"
with open(CFG_PATH, "w") as f:
    yaml.dump(cfg, f)
print(f"✓ Config written: {CFG_PATH}")

# ── 6. Auto-resume LDM from previous session ─────────────────────────────────
import glob, re

LDM_CKPT_DIR = "/kaggle/working/checkpoints/diffusion_ldm"
os.makedirs(LDM_CKPT_DIR, exist_ok=True)

for dirpath, _, filenames in os.walk("/kaggle/input"):
    pth_files = [f for f in filenames if f.endswith(".pth") and "epoch_" in f
                 and "diffusion" in dirpath.lower()]
    if pth_files:
        print(f"✓ Found previous LDM checkpoints: {dirpath}")
        for f in pth_files:
            dst = os.path.join(LDM_CKPT_DIR, f)
            if not os.path.exists(dst):
                shutil.copy2(os.path.join(dirpath, f), dst)
                print(f"  Copied: {f}")
        break

# ── 7. TRAIN LDM ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print(" PHASE 2 — Latent Diffusion Model (SD-1.5 VAE + UNet)")
print("="*60)

from train_diffusion_ldm import train_ldm

with open(CFG_PATH) as f:
    loaded_cfg = yaml.safe_load(f)

unet, sar_enc = train_ldm(loaded_cfg)
print("\n✓ LDM Training complete!")

# ── 8. Quick visual evaluation ───────────────────────────────────────────────
print("\n📁 Key files to download:")
print("  checkpoints/diffusion_ldm/best.pth")
print("  outputs/losses_diffusion_ldm.csv")
print("  outputs/diffusion_ldm_samples/  (visual samples per epoch)")

print("\n✅ SAVE VERSION NOW to preserve diffusion checkpoints!")
