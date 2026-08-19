"""
eval.py — Evaluation Script for SAR-to-EO Translation

Computes all evaluation metrics on a saved set of predictions vs. ground truth:
  - LPIPS ↓  (perceptual, primary)
  - FID   ↓  (perceptual, primary)
  - SSIM  ↑  (structural, secondary)
  - PSNR  ↑  (pixel-level, secondary)

Usage:
    # Auto-run inference then evaluate:
    python eval.py --config config.yaml --weights checkpoints/full/best.pth --split test

    # Evaluate from existing prediction directories:
    python eval.py --pred_dir outputs/preds/ --gt_dir outputs/gt/

Results saved to: outputs/metrics_{ablation}_{split}.csv
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from typing import Dict, List

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from utils.metrics import compute_metrics, compute_fid


# ---------------------------------------------------------------------------
# Load images from a directory
# ---------------------------------------------------------------------------

def load_images_from_dir(directory: str) -> List[torch.Tensor]:
    """Load all PNG images as normalised [3, H, W] tensors in [-1, 1]."""
    png_files = sorted(Path(directory).glob("*.png"))
    if not png_files:
        raise FileNotFoundError(f"No PNG files in: {directory}")

    tensors = []
    for p in png_files:
        img = Image.open(p).convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0
        t   = torch.from_numpy(arr).permute(2, 0, 1) * 2.0 - 1.0
        tensors.append(t)
    return tensors


# ---------------------------------------------------------------------------
# Run inference and save predictions + ground truth to directories
# ---------------------------------------------------------------------------

def run_inference_to_dir(
    config_path:  str,
    weights_path: str,
    split:        str,
    pred_dir:     str,
    gt_dir:       str,
    use_tta:      bool = False,
) -> None:
    """
    Load the generator, run inference on the given split, and save
    both predictions and ground-truth images to separate directories.
    Loads EMA weights if available.
    """
    from data.dataloader import get_dataloaders
    from models.generator import UNetGenerator

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m_cfg  = cfg["model"]

    G = UNetGenerator(
        in_channels   = m_cfg["input_channels"],
        out_channels  = m_cfg["output_channels"],
        base_ch       = m_cfg.get("base_ch", 64),
        use_attention = m_cfg.get("use_attention", True),
        pretrained    = False,   # weights from checkpoint
        gradient_checkpointing = m_cfg.get("gradient_checkpointing", False),
    ).to(device)

    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    if "G_ema" in ckpt:
        G.load_state_dict(ckpt["G_ema"])
        print(f"[Eval] Loaded EMA weights")
    elif "G" in ckpt:
        G.load_state_dict(ckpt["G"])
        print(f"[Eval] Loaded live weights (no EMA)")
    else:
        G.load_state_dict(ckpt)
    G.eval()

    _, val_loader, test_loader = get_dataloaders(cfg)
    loader = test_loader if split == "test" else val_loader

    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(gt_dir,   exist_ok=True)

    use_amp = device.type == "cuda"
    n_saved = 0

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc=f"Inference ({split})")):
            sar     = batch["sar"].to(device)
            real_eo = batch["eo"].to(device)

            if use_tta:
                from infer import tta_predict
                fake_eo = tta_predict(G, sar, use_amp=use_amp)
            else:
                with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                    fake_eo = G(sar)

            def save_tensor(tensor, path):
                img = tensor[0].cpu()
                img = (img + 1.0) / 2.0
                img = img.clamp(0, 1)
                img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                Image.fromarray(img).save(path)

            save_tensor(fake_eo, os.path.join(pred_dir, f"{i:05d}.png"))
            save_tensor(real_eo, os.path.join(gt_dir,   f"{i:05d}.png"))
            n_saved += 1

    print(f"[Eval] Saved {n_saved} prediction/GT pairs.")


# ---------------------------------------------------------------------------
# Evaluate from directories
# ---------------------------------------------------------------------------

def evaluate_dirs(pred_dir: str, gt_dir: str,
                  output_path: str, split: str = "test") -> Dict[str, float]:
    """Compute all metrics from two directories of PNG images."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n[Eval] Computing metrics ({split})...")
    print(f"  Predictions : {pred_dir}")
    print(f"  Ground truth: {gt_dir}")

    preds   = load_images_from_dir(pred_dir)
    targets = load_images_from_dir(gt_dir)

    if len(preds) != len(targets):
        raise ValueError(f"Count mismatch: {len(preds)} preds vs {len(targets)} GT")

    metrics = compute_metrics(preds, targets, device=device)

    print("[Eval] Computing FID...")
    metrics["fid"] = compute_fid(pred_dir, gt_dir, device=device)

    print(f"\n{'='*50}")
    print(f" Evaluation — {split}")
    print(f"{'='*50}")
    print(f"  LPIPS ↓ : {metrics['lpips']:.4f}   (primary)")
    print(f"  FID   ↓ : {metrics['fid']:.2f}   (primary)")
    print(f"  SSIM  ↑ : {metrics['ssim']:.4f}   (secondary)")
    print(f"  PSNR  ↑ : {metrics['psnr']:.2f} dB (secondary)")
    print(f"{'='*50}\n")

    import csv
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "lpips", "fid", "ssim", "psnr"])
        writer.writerow([split, metrics["lpips"], metrics["fid"],
                         metrics["ssim"],  metrics["psnr"]])
    print(f"[Eval] Results → {output_path}")
    return metrics


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

    parser = argparse.ArgumentParser(description="SAR-to-EO Evaluation")
    # Both spellings accepted throughout: hyphens are the argparse convention,
    # underscores are what this repo shipped with. argparse derives `dest` from
    # the first option string, so args.pred_dir etc. are unchanged.
    parser.add_argument("--pred-dir", "--pred_dir", type=str, default=None)
    parser.add_argument("--gt-dir",   "--gt_dir",   type=str, default=None)
    parser.add_argument("--config",    type=str, default="config.yaml")
    parser.add_argument("--weights",   type=str, default=None)
    parser.add_argument("--split",     type=str, default="test",
                        choices=["val", "test"])
    parser.add_argument("--output",    type=str, default=None)
    parser.add_argument("--tta",       action="store_true",
                        help="Use test-time augmentation during inference")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ablation = cfg.get("active_ablation", "full")

    pred_dir = args.pred_dir or os.path.join(
        cfg["paths"]["output_dir"], f"eval_preds_{ablation}_{args.split}")
    gt_dir   = args.gt_dir   or os.path.join(
        cfg["paths"]["output_dir"], f"eval_gt_{ablation}_{args.split}")
    out_csv  = args.output   or os.path.join(
        cfg["paths"]["output_dir"], f"metrics_{ablation}_{args.split}.csv")

    if not args.pred_dir or not list(Path(pred_dir).glob("*.png")):
        if not args.weights:
            print("[ERROR] Provide --pred_dir or --weights.")
            sys.exit(1)
        run_inference_to_dir(args.config, args.weights, args.split,
                             pred_dir, gt_dir, use_tta=args.tta)

    evaluate_dirs(pred_dir, gt_dir, out_csv, split=args.split)
