"""
plot_results.py — Full Results Visualization

Generates a comprehensive results figure after training:
  1. Loss curves (all components: L1, FFT, VGG, ADV, SSIM, D)
  2. Per-terrain metric bar chart (if eval_per_terrain.py was run)
  3. Best triplet samples (SAR | Generated EO | Ground Truth)
  4. Model comparison table (GAN vs Diffusion vs ControlNet)

Usage:
    python plot_results.py --config config.yaml
    python plot_results.py --config config.yaml --compare diffusion_ldm
"""

import os
import csv
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image


# ---------------------------------------------------------------------------
# Load loss CSV
# ---------------------------------------------------------------------------

def load_loss_csv(path: str) -> Dict[str, List[float]]:
    if not os.path.exists(path):
        return {}
    data = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                if k == "epoch":
                    continue
                if k not in data:
                    data[k] = []
                try:
                    data[k].append(float(v))
                except (ValueError, TypeError):
                    pass
    return data


def load_metrics_csv(path: str) -> Dict[str, float]:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            return {k: float(v) for k, v in row.items() if k != "split"}
    return {}


# ---------------------------------------------------------------------------
# Main plot function
# ---------------------------------------------------------------------------

def plot_results(cfg: dict, compare_model: Optional[str] = None):
    out_dir     = cfg["paths"]["output_dir"]
    ckpt_dir    = cfg["paths"]["checkpoint_dir"]
    os.makedirs(out_dir, exist_ok=True)

    losses_path  = os.path.join(out_dir, "losses_full.csv")
    metrics_path = os.path.join(out_dir, "metrics_test.csv")
    terrain_path = os.path.join(out_dir, "metrics_per_terrain.csv")
    samples_dir  = os.path.join(out_dir, "samples", "full")

    # ── Set matplotlib style ─────────────────────────────────────────────
    plt.rcParams.update({
        "font.family":     "DejaVu Sans",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.3,
        "figure.dpi":        150,
    })

    # ── Load data ─────────────────────────────────────────────────────────
    losses    = load_loss_csv(losses_path)
    metrics   = load_metrics_csv(metrics_path)

    # ── Figure layout ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 14))
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

    # ── Row 1: Loss curves ────────────────────────────────────────────────
    loss_keys = [
        ("G_total", "#2196F3", "Generator (total)"),
        ("G_l1",    "#4CAF50", "L1 reconstruction"),
        ("G_vgg",   "#FF9800", "VGG perceptual"),
        ("D_total", "#F44336", "Discriminator"),
    ]
    epochs = list(range(1, len(losses.get("G_total", [])) + 1))

    for col, (key, color, label) in enumerate(loss_keys):
        if key not in losses:
            continue
        ax = fig.add_subplot(gs[0, col])
        ax.plot(epochs, losses[key], color=color, lw=1.5, label=label)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel("Loss",  fontsize=9)
        if losses[key]:
            ax.annotate(
                f"Final: {losses[key][-1]:.3f}",
                xy=(epochs[-1], losses[key][-1]),
                xytext=(-30, 10), textcoords="offset points",
                fontsize=8, color=color,
                arrowprops=dict(arrowstyle="->", color=color, lw=0.8),
            )

    # ── Row 2: Overall metrics + per-terrain ──────────────────────────────
    # Metrics summary
    ax_m = fig.add_subplot(gs[1, 0])
    if metrics:
        keys   = ["ssim", "psnr", "lpips", "fid"]
        labels = ["SSIM ↑", "PSNR ↑", "LPIPS ↓", "FID ↓"]
        vals   = [metrics.get(k, 0) for k in keys]
        colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0"]
        bars   = ax_m.bar(labels, vals, color=colors, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax_m.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                      f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax_m.set_title("Overall Test Metrics", fontsize=10, fontweight="bold")
        ax_m.set_ylim(0, max(vals) * 1.2)

    # Per-terrain breakdown
    if os.path.exists(terrain_path):
        terrain_data = {}
        with open(terrain_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                terrain_data[row["terrain"]] = {
                    "ssim":  float(row["ssim"]),
                    "psnr":  float(row["psnr"]),
                    "lpips": float(row["lpips"]),
                }

        terrains = sorted(terrain_data.keys())
        t_colors = ["#4CAF50", "#FF9800", "#F44336", "#2196F3"][:len(terrains)]

        for col_idx, (metric_key, m_label) in enumerate(
            [("ssim", "SSIM ↑"), ("psnr", "PSNR (dB) ↑"), ("lpips", "LPIPS ↓")]
        ):
            if col_idx + 1 >= 4:
                break
            ax_t = fig.add_subplot(gs[1, col_idx + 1])
            vals_t = [terrain_data[t][metric_key] for t in terrains]
            ax_t.bar(terrains, vals_t, color=t_colors, edgecolor="white")
            ax_t.set_title(f"Per-Terrain {m_label}", fontsize=10, fontweight="bold")
            ax_t.tick_params(axis="x", rotation=15)
            for i, v in enumerate(vals_t):
                ax_t.text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=8)

    # ── Row 3: Sample triplets ────────────────────────────────────────────
    sample_files = []
    if os.path.exists(samples_dir):
        sample_files = sorted(Path(samples_dir).glob("epoch*_grid.png"))[-3:]
        if not sample_files:
            sample_files = sorted(Path(samples_dir).glob("epoch*_000.png"))[-3:]

    for col_idx, sf in enumerate(sample_files[:4]):
        if col_idx >= 4:
            break
        ax_s = fig.add_subplot(gs[2, col_idx])
        img  = Image.open(sf).convert("RGB")
        ax_s.imshow(np.array(img))
        ax_s.set_title(sf.stem, fontsize=9)
        ax_s.axis("off")

    # ── Title and save ────────────────────────────────────────────────────
    epoch_str = f" ({len(epochs)} epochs)" if epochs else ""
    fig.suptitle(
        f"SAR→EO Results — ResNet50-UNet GAN{epoch_str}",
        fontsize=16, fontweight="bold", y=1.01,
    )

    save_path = os.path.join(out_dir, "results_summary.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Results summary saved → {save_path}")

    # ── Print metric table ────────────────────────────────────────────────
    if metrics:
        print("\n" + "="*45)
        print("  FINAL TEST METRICS")
        print("="*45)
        print(f"  SSIM  ↑ : {metrics.get('ssim',  0):.4f}")
        print(f"  PSNR  ↑ : {metrics.get('psnr',  0):.2f} dB")
        print(f"  LPIPS ↓ : {metrics.get('lpips', 0):.4f}")
        print(f"  FID   ↓ : {metrics.get('fid',   0):.2f}")
        print("="*45)

    return save_path


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
    parser.add_argument("--config",  default="config.yaml")
    parser.add_argument("--compare", default=None,
                        help="Name of model to compare (e.g. diffusion_ldm)")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    plot_results(cfg, args.compare)
