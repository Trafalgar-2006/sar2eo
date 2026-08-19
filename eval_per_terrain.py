"""
eval_per_terrain.py — Per-Terrain Metric Breakdown

Runs evaluation separately for each terrain class (agri, barrenland, grassland, urban)
and produces a detailed comparison table. Urban is always hardest — this shows that
explicitly and makes the results analysis much richer.

Usage:
    python eval_per_terrain.py --config config.yaml --weights checkpoints/full/best.pth

Outputs:
    outputs/metrics_per_terrain.csv      — full per-terrain table
    outputs/metrics_per_terrain.png      — bar chart comparison
    outputs/triplets_per_terrain/        — sample triplets per terrain
"""

import os
import sys
import csv
import yaml
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from data.dataloader import SARtoEODataset
from models.generator import UNetGenerator
from utils.metrics import compute_ssim, compute_psnr, compute_lpips


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model(cfg: dict, weights_path: str, device: torch.device):
    m = cfg.get("model", {})
    G = UNetGenerator(
        in_channels=m.get("input_channels", 1),
        out_channels=m.get("output_channels", 3),
        base_ch=m.get("base_ch", 64),
        use_attention=m.get("use_attention", True),
        pretrained=False,
        full_res_skip=m.get("full_res_skip", True),
    ).to(device)
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    state = ckpt.get("G_ema") or ckpt.get("G") or ckpt
    G.load_state_dict(state)
    G.eval()
    return G


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """[-1,1] tensor [C,H,W] → PIL"""
    t = (t.float() + 1) / 2
    t = t.clamp(0, 1)
    arr = (t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def save_triplet(sar_t, pred_t, gt_t, path: str):
    sar_np  = np.array(tensor_to_pil(sar_t.repeat(3,1,1) if sar_t.shape[0]==1 else sar_t))
    pred_np = np.array(tensor_to_pil(pred_t))
    gt_np   = np.array(tensor_to_pil(gt_t))
    grid    = np.concatenate([sar_np, pred_np, gt_np], axis=1)
    Image.fromarray(grid).save(path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def eval_per_terrain(cfg: dict, weights_path: str):
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G       = load_model(cfg, weights_path, device)

    data_cfg = cfg["data"]
    img_size = data_cfg.get("image_size", 256)

    # ── Take the real test split, then bucket it by terrain ───────────────
    # Deliberately NOT re-derived by slicing files here: the test set must be
    # the same scene-disjoint split the model was held out from, or these
    # numbers would be measured partly on training scenes.
    test_ds = SARtoEODataset(cfg, split="test", augment=False)

    def terrain_of(sar_path: str) -> str:
        """Terrain is the directory above the s1/ folder: <root>/<terrain>/s1/x.png"""
        parts = Path(sar_path).parts
        return parts[-3] if len(parts) >= 3 else "unknown"

    by_terrain = defaultdict(list)
    for sar_path, eo_path in test_ds.pairs:
        by_terrain[terrain_of(sar_path)].append((sar_path, eo_path))

    terrain_dirs = sorted(by_terrain)
    print(f"\nTest split: {len(test_ds.pairs):,} pairs across "
          f"{len(terrain_dirs)} terrain class(es)")
    for t in terrain_dirs:
        print(f"  {t}: {len(by_terrain[t]):,} pairs")

    results = {}   # terrain → {"ssim": [...], "psnr": [...], "lpips": [...]}
    triplet_dir = os.path.join(cfg["paths"]["output_dir"], "triplets_per_terrain")
    os.makedirs(triplet_dir, exist_ok=True)

    use_amp = device.type == "cuda"
    lpips_fn = compute_lpips.__self__ if hasattr(compute_lpips, "__self__") else None

    # ── Per-terrain loop ──────────────────────────────────────────────────
    for terrain in terrain_dirs:
        pairs = by_terrain[terrain]
        print(f"\n[{terrain}] Evaluating {len(pairs)} test pairs ...")

        ssims, psnrs, lpips_vals = [], [], []
        saved_triplet = False

        for idx, (s1_path, s2_path) in enumerate(tqdm(pairs, desc=terrain, leave=False)):
            # Load and preprocess
            sar_pil = Image.open(s1_path).convert("L").resize((img_size, img_size), Image.LANCZOS)
            eo_pil  = Image.open(s2_path).convert("RGB").resize((img_size, img_size), Image.LANCZOS)

            sar_np  = np.array(sar_pil, dtype=np.float32) / 255.0
            eo_np   = np.array(eo_pil,  dtype=np.float32) / 255.0

            sar_t   = torch.from_numpy(sar_np * 2 - 1).unsqueeze(0).unsqueeze(0).to(device)
            eo_t    = torch.from_numpy((eo_np * 2 - 1).transpose(2, 0, 1)).unsqueeze(0).to(device)

            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                    pred_t = G(sar_t)

            # Metrics (on [0,1] scale)
            pred_01 = (pred_t.squeeze(0) + 1) / 2
            gt_01   = (eo_t.squeeze(0)   + 1) / 2

            ssims.append(compute_ssim(pred_01, gt_01))
            psnrs.append(compute_psnr(pred_01, gt_01))
            try:
                lpips_vals.append(compute_lpips(pred_t, eo_t))
            except Exception:
                pass

            # Save one triplet per terrain for visual comparison
            if not saved_triplet:
                triplet_path = os.path.join(triplet_dir, f"{terrain}_sample.png")
                save_triplet(sar_t.squeeze(0).cpu(), pred_t.squeeze(0).cpu(),
                             eo_t.squeeze(0).cpu(), triplet_path)
                saved_triplet = True

        results[terrain] = {
            "ssim":  float(np.mean(ssims))  if ssims  else 0.0,
            "psnr":  float(np.mean(psnrs))  if psnrs  else 0.0,
            "lpips": float(np.mean(lpips_vals)) if lpips_vals else -1.0,
            "n":     len(pairs),
        }

        print(f"  SSIM={results[terrain]['ssim']:.4f}  "
              f"PSNR={results[terrain]['psnr']:.2f}dB  "
              f"LPIPS={results[terrain]['lpips']:.4f}")

    # ── Print summary table ───────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"  PER-TERRAIN RESULTS — {weights_path.split('/')[-1]}")
    print("="*60)
    print(f"{'Terrain':<14} {'SSIM↑':>8} {'PSNR↑':>9} {'LPIPS↓':>9} {'N':>6}")
    print("-"*60)
    for t, m in sorted(results.items()):
        print(f"{t:<14} {m['ssim']:>8.4f} {m['psnr']:>8.2f}dB {m['lpips']:>9.4f} {m['n']:>6}")
    print("="*60)

    # Overall average
    all_ssim  = np.mean([v["ssim"]  for v in results.values()])
    all_psnr  = np.mean([v["psnr"]  for v in results.values()])
    all_lpips = np.mean([v["lpips"] for v in results.values() if v["lpips"] >= 0])
    print(f"{'OVERALL':<14} {all_ssim:>8.4f} {all_psnr:>8.2f}dB {all_lpips:>9.4f}")
    print("="*60)

    # ── Save CSV ──────────────────────────────────────────────────────────
    out_dir  = cfg["paths"]["output_dir"]
    csv_path = os.path.join(out_dir, "metrics_per_terrain.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["terrain","ssim","psnr","lpips","n"])
        writer.writeheader()
        for t, m in sorted(results.items()):
            writer.writerow({"terrain": t, **m})
    print(f"\n[✓] CSV saved → {csv_path}")

    # ── Bar chart ─────────────────────────────────────────────────────────
    terrains = sorted(results.keys())
    ssims    = [results[t]["ssim"]  for t in terrains]
    psnrs    = [results[t]["psnr"]  for t in terrains]
    lpips_v  = [results[t]["lpips"] for t in terrains]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    colors = ["#4CAF50", "#FF9800", "#F44336", "#2196F3"][:len(terrains)]

    for ax, vals, label, better in zip(
        axes,
        [ssims, psnrs, lpips_v],
        ["SSIM ↑", "PSNR ↑ (dB)", "LPIPS ↓"],
        ["Higher", "Higher", "Lower"]
    ):
        bars = ax.bar(terrains, vals, color=colors, edgecolor="white", linewidth=1.2)
        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.set_ylabel(f"{label} ({better} is better)", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        # Value labels on bars
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        ax.tick_params(axis="x", rotation=15)

    fig.suptitle("SAR→EO Per-Terrain Metric Breakdown", fontsize=14, fontweight="bold")
    plt.tight_layout()
    png_path = os.path.join(out_dir, "metrics_per_terrain.png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Chart saved → {png_path}")
    print(f"[✓] Triplets saved → {triplet_dir}/")

    return results


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

    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="config.yaml")
    parser.add_argument("--weights", default="checkpoints/full/best.pth")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    eval_per_terrain(cfg, args.weights)
