"""
eval.py — Evaluation Script for SAR-to-EO Translation

Computes all required metrics on a saved set of predictions vs. ground truth:
  - LPIPS ↓  (primary)
  - FID   ↓  (primary, directory-based)
  - SSIM  ↑  (secondary)
  - PSNR  ↑  (secondary)

Usage:
    python eval.py --pred_dir <path/to/generated_eo> --gt_dir <path/to/ground_truth_eo>

Or run full inference + evaluation together:
    python eval.py --config config.yaml --weights checkpoints/full/best.pth --split test

The script saves a results table to outputs/metrics_{split}.csv.
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

def load_images_from_dir(directory: str, device: str = "cpu") -> List[torch.Tensor]:
    """
    Load all PNG images from a directory as normalised [3, H, W] tensors in [-1, 1].
    Sorted by filename for deterministic ordering.
    """
    png_files = sorted(Path(directory).glob("*.png"))
    if not png_files:
        raise FileNotFoundError(f"No PNG files found in: {directory}")

    tensors = []
    for p in png_files:
        img = Image.open(p).convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0   # [0, 1]
        t   = torch.from_numpy(arr).permute(2, 0, 1)      # [3, H, W]
        t   = t * 2.0 - 1.0                               # [-1, 1]
        tensors.append(t)

    return tensors


# ---------------------------------------------------------------------------
# Run inference using Generator + save predictions to directory
# ---------------------------------------------------------------------------

def run_inference_to_dir(
    config_path: str,
    weights_path: str,
    split: str,
    pred_dir: str,
    gt_dir: str,
) -> None:
    """
    Load the generator, run inference on the given split, and save
    both predictions and ground-truth images to separate directories.
    Used when --pred_dir/--gt_dir are not provided (auto-run mode).
    """
    import yaml
    from data.dataloader import get_dataloaders
    from models.generator import UNetGenerator

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model_cfg = cfg["model"]
    G = UNetGenerator(
        in_channels  = model_cfg["input_channels"],
        out_channels = model_cfg["output_channels"],
        base_ch      = model_cfg["base_channels"],
    ).to(device)

    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    G.load_state_dict(ckpt["G"])
    G.eval()
    print(f"[Eval] Loaded weights from {weights_path}")

    _, val_loader, test_loader = get_dataloaders(cfg)
    loader = test_loader if split == "test" else val_loader

    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(gt_dir,   exist_ok=True)
    pred_dir_path = Path(pred_dir)

    n_saved = 0
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc=f"Running inference ({split})")):
            sar     = batch["sar"].to(device)
            real_eo = batch["eo"].to(device)
            fake_eo = G(sar)

            # Save prediction
            pred_img = fake_eo[0].cpu()
            pred_img = (pred_img + 1.0) / 2.0
            pred_img = pred_img.clamp(0, 1)
            pred_img = (pred_img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            Image.fromarray(pred_img).save(os.path.join(pred_dir, f"{i:05d}.png"))

            # Save ground truth
            gt_img = real_eo[0].cpu()
            gt_img = (gt_img + 1.0) / 2.0
            gt_img = gt_img.clamp(0, 1)
            gt_img = (gt_img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            Image.fromarray(gt_img).save(os.path.join(gt_dir, f"{i:05d}.png"))
            n_saved += 1

    n_saved = sum(1 for _ in pred_dir_path.glob("*.png")) if hasattr(pred_dir_path, 'glob') else i+1
    print(f"[Eval] Saved prediction/GT pairs.")


# ---------------------------------------------------------------------------
# Evaluate from directories
# ---------------------------------------------------------------------------

def evaluate_dirs(pred_dir: str, gt_dir: str,
                  output_path: str, split: str = "test") -> Dict[str, float]:
    """
    Compute all metrics from two directories of PNG images.

    Args:
        pred_dir:    Directory with generated EO images
        gt_dir:      Directory with ground-truth EO images
        output_path: Path to save metrics CSV
        split:       'val' or 'test' (for logging only)
    Returns:
        Dict of metric name → value
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n[Eval] Computing metrics on {split} split...")
    print(f"  Predictions : {pred_dir}")
    print(f"  Ground truth: {gt_dir}")

    preds   = load_images_from_dir(pred_dir)
    targets = load_images_from_dir(gt_dir)

    assert len(preds) == len(targets), (
        f"Mismatch: {len(preds)} predictions vs {len(targets)} ground truths"
    )

    # Per-image LPIPS, SSIM, PSNR
    metrics = compute_metrics(preds, targets, device=device)

    # FID (directory-based)
    print("[Eval] Computing FID (may take a minute)...")
    fid = compute_fid(pred_dir, gt_dir, device=device)
    metrics["fid"] = fid

    # Print results table
    print(f"\n{'='*45}")
    print(f" Evaluation Results — {split}")
    print(f"{'='*45}")
    print(f"  LPIPS ↓ : {metrics['lpips']:.4f}   (primary)")
    print(f"  FID   ↓ : {metrics['fid']:.2f}   (primary)")
    print(f"  SSIM  ↑ : {metrics['ssim']:.4f}   (secondary)")
    print(f"  PSNR  ↑ : {metrics['psnr']:.2f} dB (secondary)")
    print(f"{'='*45}\n")

    # Save to CSV
    import csv
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "lpips", "fid", "ssim", "psnr"])
        writer.writerow([split, metrics["lpips"], metrics["fid"],
                         metrics["ssim"], metrics["psnr"]])
    print(f"[Eval] Results saved -> {output_path}")

    return metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAR-to-EO Evaluation")

    # Mode 1: evaluate from existing directories
    parser.add_argument("--pred_dir",  type=str, default=None,
                        help="Directory of generated EO PNG images")
    parser.add_argument("--gt_dir",    type=str, default=None,
                        help="Directory of ground-truth EO PNG images")

    # Mode 2: auto-run inference then evaluate
    parser.add_argument("--config",    type=str, default="config.yaml",
                        help="Path to config YAML (used for auto-inference mode)")
    parser.add_argument("--weights",   type=str, default=None,
                        help="Path to model checkpoint (used for auto-inference mode)")
    parser.add_argument("--split",     type=str, default="test",
                        choices=["val", "test"],
                        help="Which split to evaluate")

    parser.add_argument("--output",    type=str, default=None,
                        help="Path to save metrics CSV")

    args = parser.parse_args()

    # Load config to get ablation name and output paths
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    ablation = cfg.get("active_ablation", "full")

    # Auto-generate output paths if not provided
    pred_dir = args.pred_dir or os.path.join(
        cfg["paths"]["output_dir"], f"eval_preds_{ablation}_{args.split}")
    gt_dir   = args.gt_dir   or os.path.join(
        cfg["paths"]["output_dir"], f"eval_gt_{ablation}_{args.split}")
    out_csv  = args.output   or os.path.join(
        cfg["paths"]["output_dir"], f"metrics_{ablation}_{args.split}.csv")

    # If pred_dir doesn't exist or is empty, run inference first
    if not args.pred_dir or not list(Path(pred_dir).glob("*.png")):
        if not args.weights:
            print("[ERROR] Provide --pred_dir or --weights to run inference.")
            sys.exit(1)
        run_inference_to_dir(args.config, args.weights, args.split, pred_dir, gt_dir)

    evaluate_dirs(pred_dir, gt_dir, out_csv, split=args.split)
