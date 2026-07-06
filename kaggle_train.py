"""
SINGLE CELL — SAR2EO Complete Training on Kaggle
Paste this entire block into one Kaggle cell and run.
"""

import subprocess, os, sys, yaml, shutil, torch

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

# ── 2. Install extra deps ─────────────────────────────────────────────────────
subprocess.run("pip install -q lpips pytorch-fid", shell=True, check=True)
print("✓ Deps installed")

# ── 2.5 Auto-resume: restore checkpoints from previous Kaggle session ─────────
# If you add a previous session's output as an input dataset, this copies
# the checkpoints so training auto-resumes from the last saved epoch.
CKPT_DST = "/kaggle/working/checkpoints/full"
os.makedirs(CKPT_DST, exist_ok=True)

resumed = False
for dirpath, dirnames, filenames in os.walk("/kaggle/input"):
    pth_files = [f for f in filenames if f.endswith(".pth")]
    if pth_files:
        print(f"✓ Found previous checkpoints in: {dirpath}")
        for f in pth_files:
            src = os.path.join(dirpath, f)
            dst = os.path.join(CKPT_DST, f)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
                print(f"  Copied: {f}")
        resumed = True
        break

if resumed:
    latest = sorted([f for f in os.listdir(CKPT_DST) if f.startswith("epoch_")])
    print(f"  Will resume from: {latest[-1] if latest else 'best.pth'}")
else:
    print("  No previous checkpoints found — starting from scratch")

# ── 3. Auto-discover dataset path (searches ANY depth) ───────────────────────
INPUT_ROOT  = "/kaggle/input"
KAGGLE_DATA = None
TERRAIN_KEYS = {"agri", "urban", "grassland", "barrenland",
                "forest", "water", "mountain"}

print(f"Scanning {INPUT_ROOT} for terrain dataset...")
for dirpath, dirnames, _ in os.walk(INPUT_ROOT):
    subdirs = {d.lower() for d in dirnames}
    if len(subdirs & TERRAIN_KEYS) >= 2:   # ≥2 terrain folders = it's the dataset root
        KAGGLE_DATA = dirpath
        print(f"✓ Dataset found at: {KAGGLE_DATA}")
        print(f"  Terrain folders : {sorted(subdirs & TERRAIN_KEYS)}")
        break

if KAGGLE_DATA is None:
    # Print full tree so user knows exactly what's mounted
    print("\n❌ Could not find terrain dataset. Full /kaggle/input tree:")
    for dirpath, dirnames, _ in os.walk(INPUT_ROOT):
        depth = dirpath.replace(INPUT_ROOT, "").count(os.sep)
        if depth > 4:
            continue
        indent = "  " * depth
        print(f"{indent}{os.path.basename(dirpath)}/")
    raise FileNotFoundError(
        "Terrain dataset not found. Go to Kaggle Notebook → Add Input → "
        "search 'sentinel12-image-pairs-segregated-by-terrain' and add it."
    )

# ── 4. Verify GPU ─────────────────────────────────────────────────────────
print(f"\n✓ CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

# ── 5. Write Kaggle-specific config ──────────────────────────────────────
config = {
    "model": {
        "input_channels": 1,
        "output_channels": 3,
        "base_ch": 64,
        "use_attention": True,
        "pretrained_encoder": True,
        "gradient_checkpointing": False,
        "n_scales_D": 3,
        "n_layers_D": 3,
    },
    "training": {
        "epochs": 65,           # fits in 12hr Kaggle session (65 × ~9.5min ≈ 10.5hrs)
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
        "save_freq": 5,         # save every 5 epochs — more recovery points
        "val_freq": 10,         # validate every 10 epochs — saves ~1hr vs val_freq=5
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
        "dataset_type":   "kaggle",
        "split_strategy": "random",
        "sen12_root":     "./data/SEN1-2",
        "train_seasons":  ["spring", "summer", "fall"],
        "val_seasons":    ["winter"],
        "test_seasons":   ["winter"],
        "kaggle_root":    KAGGLE_DATA,
        "train_terrain":  ["agri", "barrenland", "grassland"],
        "val_terrain":    ["urban"],
        "test_terrain":   ["urban"],
        "image_size":     256,
        "subset_size":    None,
        "num_workers":    2,
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

CFG_PATH = f"{REPO}/config_kaggle.yaml"
with open(CFG_PATH, "w") as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
print(f"✓ Config written → {CFG_PATH}")

# ── 6. Quick smoke test ───────────────────────────────────────────────────
from models.generator     import UNetGenerator
from models.discriminator import MultiScaleDiscriminator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G = UNetGenerator(in_channels=1, out_channels=3, use_attention=True, pretrained=True).to(device)
D = MultiScaleDiscriminator().to(device)

with torch.no_grad():
    with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
        out  = G(torch.randn(2, 1, 256, 256).to(device))
        disc = D(torch.randn(2, 1, 256, 256).to(device),
                 torch.randn(2, 3, 256, 256).to(device))

print(f"✓ G output: {out.shape}, range=[{out.min():.2f},{out.max():.2f}]")
print(f"✓ D scales: {[tuple(d.shape) for d in disc]}")
print(f"✓ G params: {sum(p.numel() for p in G.parameters()):,}")
vram_used = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
print(f"✓ VRAM (smoke test): {vram_used:.2f} GB")
del G, D, out, disc
torch.cuda.empty_cache()

# ── 7. TRAIN ─────────────────────────────────────────────────────────────
from train import train, load_config, make_dirs

cfg = load_config(CFG_PATH)
make_dirs(cfg)

print("\n" + "="*60)
print(" TRAINING — ResNet50-UNet + CBAM + Multi-Scale PatchGAN")
print("="*60 + "\n")

G = train(cfg)
print("\n✓ Training complete!")

# ── 8. EVALUATE ──────────────────────────────────────────────────────────
from eval import run_inference_to_dir, evaluate_dirs

WEIGHTS   = "/kaggle/working/checkpoints/full/best.pth"
PRED_DIR  = "/kaggle/working/outputs/eval_preds_test"
GT_DIR    = "/kaggle/working/outputs/eval_gt_test"
OUT_CSV   = "/kaggle/working/outputs/metrics_test.csv"

run_inference_to_dir(CFG_PATH, WEIGHTS, "test", PRED_DIR, GT_DIR, use_tta=False)
metrics = evaluate_dirs(PRED_DIR, GT_DIR, OUT_CSV, split="test")

print("\n" + "="*50)
print("  FINAL METRICS")
print("="*50)
print(f"  SSIM  ↑ : {metrics['ssim']:.4f}")
print(f"  PSNR  ↑ : {metrics['psnr']:.2f} dB")
print(f"  LPIPS ↓ : {metrics['lpips']:.4f}")
print(f"  FID   ↓ : {metrics['fid']:.2f}")
print("="*50)
print("\n✓ Done! Download outputs from /kaggle/working/")
