"""
kaggle_train.py — SAR-to-EO Training Script for Kaggle

Run this file directly on Kaggle instead of using the notebook.
It handles all setup, dataset discovery, and training in one shot.

How to use on Kaggle:
  1. Create a new Notebook (GPU P100 or T4)
  2. Add dataset: "requiemonk/sentinel12-image-pairs-segregated-by-terrain"
  3. Enable Internet in Settings
  4. Paste and run these cells (or upload this file and run: !python kaggle_train.py)
"""

# ============================================================
# CELL 1 — Install dependencies & clone repo
# ============================================================
import subprocess, os, sys

def run(cmd):
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")

# Install pinned dependencies
run("pip install -q lpips pytorch-fid")

# Clone latest code
if os.path.exists("/kaggle/working/sar2eo"):
    run("cd /kaggle/working/sar2eo && git pull")
else:
    run("git clone https://github.com/Trafalgar-2006/sar2eo.git /kaggle/working/sar2eo")

os.chdir("/kaggle/working/sar2eo")
sys.path.insert(0, "/kaggle/working/sar2eo")
print("\n✓ Repo ready at /kaggle/working/sar2eo")

# ============================================================
# CELL 2 — Verify GPU + dataset
# ============================================================
import torch

print(f"PyTorch : {torch.__version__}")
print(f"CUDA    : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU     : {torch.cuda.get_device_name(0)}")
    print(f"VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Check Kaggle dataset is mounted
KAGGLE_DATA = "/kaggle/input/sentinel12-image-pairs-segregated-by-terrain"
SEN12_DATA  = "/kaggle/input/sen1-2"          # Optional — if you added SEN1-2 too

if os.path.exists(KAGGLE_DATA):
    terrains = [d for d in os.listdir(KAGGLE_DATA)
                if os.path.isdir(os.path.join(KAGGLE_DATA, d))]
    print(f"\n✓ Kaggle dataset found: {terrains}")
else:
    raise FileNotFoundError(
        f"Dataset not found at {KAGGLE_DATA}\n"
        f"Add dataset 'requiemonk/sentinel12-image-pairs-segregated-by-terrain' "
        f"in Kaggle Notebook settings."
    )

if os.path.exists(SEN12_DATA):
    print(f"✓ SEN1-2 dataset found at {SEN12_DATA}")
else:
    print(f"ℹ SEN1-2 not found — training on Kaggle dataset only (still great)")
    SEN12_DATA = None

# ============================================================
# CELL 3 — Write config for this environment
# ============================================================
import yaml

# Determine dataset mode
dataset_type = "combined" if SEN12_DATA else "kaggle"

config = {
    "model": {
        "input_channels": 1,
        "output_channels": 3,
        "base_ch": 64,
        "use_attention": True,
        "pretrained_encoder": True,
        "gradient_checkpointing": False,  # set True if you get OOM
        "n_scales_D": 3,
        "n_layers_D": 3,
    },
    "training": {
        "epochs": 150,
        "batch_size": 8,
        "lr_encoder": 2e-5,
        "lr_decoder": 2e-4,
        "lr_discriminator": 2e-4,
        "beta1": 0.5,
        "beta2": 0.999,
        "warmup_epochs": 5,
        "lr_min": 1e-6,
        "gradient_clip_norm": 1.0,
        "ema_decay": 0.999,
        "mixed_precision": True,
        "save_freq": 10,
        "val_freq": 5,
        "seed": 42,
    },
    "loss": {
        "lambda_l1":   100.0,
        "lambda_adv":    1.0,
        "lambda_fft":   10.0,
        "lambda_vgg":   10.0,
        "lambda_ssim":   5.0,
    },
    "active_ablation": "full",
    "data": {
        "dataset_type":   dataset_type,
        "split_strategy": "random",
        "sen12_root":     SEN12_DATA or "./data/SEN1-2",
        "train_seasons":  ["spring", "summer", "fall"],
        "val_seasons":    ["winter"],
        "test_seasons":   ["winter"],
        "kaggle_root":    KAGGLE_DATA,
        "train_terrain":  ["agri", "barrenland", "grassland"],
        "val_terrain":    ["urban"],
        "test_terrain":   ["urban"],
        "image_size":     256,
        "subset_size":    None,
        "num_workers":    2,         # 2 is safe on Kaggle
    },
    "augmentation": {
        "horizontal_flip":      True,
        "vertical_flip":        True,
        "rotation_90":          True,
        "sar_gaussian_noise":   True,
        "eo_brightness_jitter": True,
    },
    "paths": {
        "checkpoint_dir": "/kaggle/working/checkpoints",
        "output_dir":     "/kaggle/working/outputs",
        "log_dir":        "/kaggle/working/logs",
    },
}

config_path = "/kaggle/working/sar2eo/config_kaggle.yaml"
with open(config_path, "w") as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print(f"✓ Config written → {config_path}")
print(f"  dataset_type  : {dataset_type}")
print(f"  epochs        : {config['training']['epochs']}")
print(f"  batch_size    : {config['training']['batch_size']}")
print(f"  mixed_precision: {config['training']['mixed_precision']}")

# ============================================================
# CELL 4 — Quick model smoke test (shape + VRAM)
# ============================================================
import torch
from models.generator     import UNetGenerator
from models.discriminator import MultiScaleDiscriminator

device = torch.device("cuda")
G = UNetGenerator(in_channels=1, out_channels=3, use_attention=True, pretrained=True).to(device)
D = MultiScaleDiscriminator().to(device)

x    = torch.randn(2, 1, 256, 256).to(device)
eo   = torch.randn(2, 3, 256, 256).to(device)

with torch.amp.autocast(device_type="cuda"):
    out  = G(x)
    disc = D(x, eo)

print(f"G output : {out.shape}    range=[{out.min():.2f},{out.max():.2f}]")
print(f"D scales : {[tuple(d.shape) for d in disc]}")
print(f"G params : {sum(p.numel() for p in G.parameters()):,}")
vram = torch.cuda.max_memory_allocated() / 1e9
print(f"VRAM used: {vram:.2f} GB")

# If VRAM > 14 GB, enable gradient_checkpointing in config
if vram > 14:
    print("⚠ VRAM tight — consider setting gradient_checkpointing: true")

del G, D, x, eo, out
torch.cuda.empty_cache()
print("\n✓ Smoke test passed")

# ============================================================
# CELL 5 — TRAIN (the main event)
# ============================================================
import sys
sys.argv = ["train.py", "--config", config_path, "--ablation", "full"]

from train import train, load_config, make_dirs

cfg = load_config(config_path)
make_dirs(cfg)

print("\n" + "="*65)
print(" Starting training — ResNet50-UNet + CBAM + Multi-Scale D")
print("="*65)
G = train(cfg)
print("\n✓ Training complete!")

# ============================================================
# CELL 6 — EVALUATE
# ============================================================
import sys
sys.argv = [
    "eval.py",
    "--config",  config_path,
    "--weights", "/kaggle/working/checkpoints/full/best.pth",
    "--split",   "test",
]

from eval import run_inference_to_dir, evaluate_dirs
import yaml

with open(config_path) as f:
    cfg = yaml.safe_load(f)

ablation  = cfg.get("active_ablation", "full")
pred_dir  = f"/kaggle/working/outputs/eval_preds_{ablation}_test"
gt_dir    = f"/kaggle/working/outputs/eval_gt_{ablation}_test"
out_csv   = f"/kaggle/working/outputs/metrics_{ablation}_test.csv"

run_inference_to_dir(
    config_path  = config_path,
    weights_path = "/kaggle/working/checkpoints/full/best.pth",
    split        = "test",
    pred_dir     = pred_dir,
    gt_dir       = gt_dir,
    use_tta      = False,     # set True for best quality (4× slower)
)

metrics = evaluate_dirs(pred_dir, gt_dir, out_csv, split="test")

# Also run with TTA and compare
print("\n--- Running TTA evaluation ---")
pred_dir_tta = pred_dir + "_tta"
gt_dir_tta   = gt_dir   + "_tta"
out_csv_tta  = out_csv.replace(".csv", "_tta.csv")

run_inference_to_dir(
    config_path  = config_path,
    weights_path = "/kaggle/working/checkpoints/full/best.pth",
    split        = "test",
    pred_dir     = pred_dir_tta,
    gt_dir       = gt_dir_tta,
    use_tta      = True,
)
metrics_tta = evaluate_dirs(pred_dir_tta, gt_dir_tta, out_csv_tta, split="test")

print("\n=== FINAL COMPARISON ===")
print(f"                   No TTA       With TTA")
print(f"  SSIM  ↑  :  {metrics['ssim']:.4f}      {metrics_tta['ssim']:.4f}")
print(f"  PSNR  ↑  :  {metrics['psnr']:.2f} dB   {metrics_tta['psnr']:.2f} dB")
print(f"  LPIPS ↓  :  {metrics['lpips']:.4f}      {metrics_tta['lpips']:.4f}")
print(f"  FID   ↓  :  {metrics['fid']:.2f}       {metrics_tta['fid']:.2f}")

# ============================================================
# CELL 7 — SAVE OUTPUTS (copy to /kaggle/working for download)
# ============================================================
import shutil

# Package everything worth keeping
artifacts = [
    "/kaggle/working/checkpoints/full/best.pth",
    "/kaggle/working/checkpoints/full/final.pth",
    out_csv,
    out_csv_tta,
    "/kaggle/working/outputs/loss_curve_full.png",
    "/kaggle/working/outputs/losses_full.csv",
]

for path in artifacts:
    if os.path.exists(path):
        print(f"✓ {path}")
    else:
        print(f"✗ NOT FOUND: {path}")

print("\nDownload these files from the Kaggle output panel →")
print("  best.pth      — trained model weights (EMA)")
print("  metrics_*.csv — evaluation results")
print("  loss_curve_full.png — training curves")
print("  losses_full.csv     — raw loss values for plotting")
