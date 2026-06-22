"""
visualize.py — Visualisation Utilities

Functions:
  - plot_loss_curves()  Save training/validation loss curves as PNG + CSV
  - save_triplets()     Save SAR input → generated EO → ground-truth EO triplets
"""

import os
import csv
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend (works on Colab/Kaggle/servers)
import matplotlib.pyplot as plt

import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Tensor → display image
# ---------------------------------------------------------------------------

def _tensor_to_rgb(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert [C, H, W] tensor from [-1, 1] to uint8 RGB numpy array.
    Handles both 1-channel (SAR → replicated to RGB) and 3-channel (EO).
    """
    img = tensor.detach().cpu().float()
    img = (img + 1.0) / 2.0
    img = img.clamp(0.0, 1.0)

    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)   # Grayscale SAR → fake-RGB for display

    img = img.permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    return img


# ---------------------------------------------------------------------------
# Loss curves
# ---------------------------------------------------------------------------

def plot_loss_curves(
    loss_history: Dict[str, List[float]],
    output_dir: str,
    ablation_name: str = "full",
) -> None:
    """
    Save training and validation loss curves.

    For GAN models: plots G_total, G_l1, G_fft, G_vgg, G_adv, D_total.
    Also saves raw values to CSV for reproducibility.

    Args:
        loss_history: Dict mapping loss name → list of per-epoch values
        output_dir:   Directory to save plot and CSV
        ablation_name: Used in filenames (e.g. "full", "l1_adv")
    """
    os.makedirs(output_dir, exist_ok=True)
    epochs = list(range(1, len(next(iter(loss_history.values()))) + 1))

    # ---- PNG plot --------------------------------------------------------
    n_plots = len(loss_history)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    colors = {
        "G_total":   "#2196F3",
        "G_l1":      "#4CAF50",
        "G_adv":     "#FF5722",
        "G_fft":     "#9C27B0",
        "G_vgg":     "#FF9800",
        "D_total":   "#F44336",
        "val_G_total": "#1565C0",
        "val_D_total": "#B71C1C",
    }

    for ax, (name, values) in zip(axes, loss_history.items()):
        color = colors.get(name, "#607D8B")
        ax.plot(epochs, values, color=color, linewidth=1.5, label=name)
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Loss", fontsize=10)
        ax.set_title(name, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle(f"Training Curves — {ablation_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()

    plot_path = os.path.join(output_dir, f"loss_curve_{ablation_name}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualize] Loss curve saved → {plot_path}")

    # ---- CSV (raw values) ------------------------------------------------
    csv_path = os.path.join(output_dir, f"losses_{ablation_name}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + list(loss_history.keys()))
        for i, epoch in enumerate(epochs):
            row = [epoch] + [loss_history[k][i] for k in loss_history]
            writer.writerow(row)
    print(f"[Visualize] Loss CSV saved → {csv_path}")


# ---------------------------------------------------------------------------
# Qualitative triplets
# ---------------------------------------------------------------------------

def save_triplets(
    sar_images:   List[torch.Tensor],
    pred_images:  List[torch.Tensor],
    gt_images:    List[torch.Tensor],
    output_dir:   str,
    prefix:       str = "triplet",
    max_triplets: int = 10,
) -> None:
    """
    Save side-by-side triplet figures: SAR input | Generated EO | Ground Truth EO.

    Saves both individual PNG per triplet and a combined summary grid.

    Args:
        sar_images:   List of [1, H, W] SAR tensors
        pred_images:  List of [3, H, W] generated EO tensors
        gt_images:    List of [3, H, W] ground-truth EO tensors
        output_dir:   Directory to save triplets
        prefix:       Filename prefix
        max_triplets: Maximum number of triplets to save
    """
    os.makedirs(output_dir, exist_ok=True)
    n = min(len(sar_images), max_triplets)

    # ---- Individual triplets ---------------------------------------------
    for i in range(n):
        sar_rgb  = _tensor_to_rgb(sar_images[i])
        pred_rgb = _tensor_to_rgb(pred_images[i])
        gt_rgb   = _tensor_to_rgb(gt_images[i])

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(sar_rgb, cmap="gray")
        axes[0].set_title("SAR Input (VV)", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(pred_rgb)
        axes[1].set_title("Generated EO", fontsize=11, fontweight="bold", color="#1565C0")
        axes[1].axis("off")

        axes[2].imshow(gt_rgb)
        axes[2].set_title("Ground Truth EO", fontsize=11, fontweight="bold", color="#1B5E20")
        axes[2].axis("off")

        plt.tight_layout()
        save_path = os.path.join(output_dir, f"{prefix}_{i+1:03d}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ---- Summary grid (all triplets in one figure) -----------------------
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = [axes]

    col_titles = ["SAR Input (VV)", "Generated EO", "Ground Truth EO"]
    for col_idx, title in enumerate(col_titles):
        axes[0][col_idx].set_title(title, fontsize=12, fontweight="bold",
                                    pad=10)

    for i in range(n):
        sar_rgb  = _tensor_to_rgb(sar_images[i])
        pred_rgb = _tensor_to_rgb(pred_images[i])
        gt_rgb   = _tensor_to_rgb(gt_images[i])

        for col_idx, img in enumerate([sar_rgb, pred_rgb, gt_rgb]):
            ax = axes[i][col_idx]
            ax.imshow(img, cmap="gray" if col_idx == 0 else None)
            ax.axis("off")

    fig.suptitle("SAR → Generated EO → Ground Truth EO", fontsize=14,
                 fontweight="bold", y=1.01)
    plt.tight_layout()

    grid_path = os.path.join(output_dir, f"{prefix}_grid.png")
    plt.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualize] {n} triplets saved → {output_dir}/")
    print(f"[Visualize] Summary grid → {grid_path}")


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs("outputs/test_viz", exist_ok=True)

    # Fake loss curves
    history = {
        "G_total": [float(1.5 - i * 0.01) for i in range(50)],
        "D_total": [float(0.8 + 0.002 * i) for i in range(50)],
    }
    plot_loss_curves(history, "outputs/test_viz", ablation_name="test")

    # Fake triplets
    sars  = [torch.randn(1, 256, 256) for _ in range(3)]
    preds = [torch.randn(3, 256, 256) for _ in range(3)]
    gts   = [torch.randn(3, 256, 256) for _ in range(3)]
    save_triplets(sars, preds, gts, "outputs/test_viz", prefix="test_triplet")

    print("Visualize test complete.")
