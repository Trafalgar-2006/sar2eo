"""
metrics.py — Evaluation Metrics for SAR-to-EO Translation

Computes all four metrics required by the assignment:

Primary (perceptual — drive ranking):
  - LPIPS ↓  Learned Perceptual Image Patch Similarity
  - FID   ↓  Fréchet Inception Distance (over full set)

Secondary (pixel-level — report and discuss):
  - SSIM  ↑  Structural Similarity Index
  - PSNR  ↑  Peak Signal-to-Noise Ratio

All metrics expect images in range [0, 1] (converted internally from [-1, 1]).
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple

from skimage.metrics import structural_similarity as sk_ssim
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
import lpips


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_numpy_uint8(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a [C, H, W] tensor from [-1, 1] to a uint8 numpy array [H, W, C].
    """
    img = tensor.detach().cpu().float()
    img = (img + 1.0) / 2.0          # [-1, 1] → [0, 1]
    img = img.clamp(0.0, 1.0)
    img = img.permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    return img                         # [H, W, C], uint8


def _to_01_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Convert [B, C, H, W] from [-1, 1] to [0, 1]."""
    return (tensor.clamp(-1, 1) + 1.0) / 2.0


# ---------------------------------------------------------------------------
# Per-image SSIM and PSNR
# ---------------------------------------------------------------------------

def _ssim_single(pred: torch.Tensor, target: torch.Tensor) -> float:
    """SSIM for a single [C, H, W] pair, in [0, 1] range."""
    p = _to_numpy_uint8(pred)    / 255.0   # [H, W, C] float
    t = _to_numpy_uint8(target)  / 255.0
    return sk_ssim(p, t, data_range=1.0, channel_axis=2)


def _psnr_single(pred: torch.Tensor, target: torch.Tensor) -> float:
    """PSNR for a single [C, H, W] pair, in [0, 1] range."""
    p = _to_numpy_uint8(pred)   / 255.0
    t = _to_numpy_uint8(target) / 255.0
    return sk_psnr(t, p, data_range=1.0)


# ---------------------------------------------------------------------------
# LPIPS (batch-level, uses pretrained network)
# ---------------------------------------------------------------------------

_lpips_fn = None

def _get_lpips(device: str) -> lpips.LPIPS:
    global _lpips_fn
    if _lpips_fn is None:
        _lpips_fn = lpips.LPIPS(net="alex").to(device)
        _lpips_fn.eval()
    return _lpips_fn


def compute_lpips(preds: torch.Tensor, targets: torch.Tensor,
                  device: str = "cpu") -> float:
    """
    Compute mean LPIPS over a batch.
    Args:
        preds:   [B, 3, H, W], range [-1, 1]
        targets: [B, 3, H, W], range [-1, 1]
    Returns:
        mean LPIPS (float, lower is better)
    """
    fn = _get_lpips(device)
    with torch.no_grad():
        scores = fn(preds.to(device), targets.to(device))
    return scores.mean().item()


# ---------------------------------------------------------------------------
# FID (Fréchet Inception Distance)
# ---------------------------------------------------------------------------

def compute_fid(pred_dir: str, gt_dir: str, device: str = "cpu") -> float:
    """
    Compute FID between two directories of PNG images using pytorch-fid.

    Args:
        pred_dir: Directory of generated EO images (256×256 RGB PNGs)
        gt_dir:   Directory of ground-truth EO images
        device:   "cuda" or "cpu"
    Returns:
        FID score (float, lower is better)
    """
    try:
        from pytorch_fid import fid_score
        fid = fid_score.calculate_fid_given_paths(
            [pred_dir, gt_dir],
            batch_size=50,
            device=device,
            dims=2048,
        )
        return fid
    except ImportError:
        print("[WARNING] pytorch-fid not installed. FID not computed.")
        return float("nan")


# ---------------------------------------------------------------------------
# compute_metrics — main entry point
# ---------------------------------------------------------------------------

def compute_metrics(
    preds: List[torch.Tensor],
    targets: List[torch.Tensor],
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Compute LPIPS, SSIM, and PSNR over lists of single-image tensors.
    FID is computed separately via compute_fid() since it needs directories.

    Args:
        preds:   List of [3, H, W] tensors, range [-1, 1]
        targets: List of [3, H, W] tensors, range [-1, 1]
        device:  Device for LPIPS computation
    Returns:
        Dict with keys: "lpips", "ssim", "psnr"
    """
    assert len(preds) == len(targets), "preds and targets must have same length"

    lpips_scores = []
    ssim_scores  = []
    psnr_scores  = []

    for pred, target in zip(preds, targets):
        # SSIM and PSNR (numpy-based, per image)
        ssim_scores.append(_ssim_single(pred, target))
        psnr_scores.append(_psnr_single(pred, target))

        # LPIPS (batch of 1)
        lp = compute_lpips(pred.unsqueeze(0), target.unsqueeze(0), device=device)
        lpips_scores.append(lp)

    return {
        "lpips": float(np.mean(lpips_scores)),
        "ssim":  float(np.mean(ssim_scores)),
        "psnr":  float(np.mean(psnr_scores)),
    }


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import torch

    # Random fake batch
    preds   = [torch.randn(3, 256, 256) for _ in range(5)]
    targets = [torch.randn(3, 256, 256) for _ in range(5)]

    results = compute_metrics(preds, targets, device="cpu")
    print("Metrics (random baseline — expect bad numbers):")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")
