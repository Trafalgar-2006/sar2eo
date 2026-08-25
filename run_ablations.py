"""
run_ablations.py — Sequential 4-config ablation study on a dedicated GPU

Trains each loss-stack configuration under IDENTICAL conditions, evaluates each
on the same held-out test split, and emits a comparison table.

    Config A  l1_only      L1
    Config B  l1_adv       L1 + multi-scale adversarial
    Config C  l1_adv_fft   L1 + adversarial + FFT
    Config D  full         L1 + adversarial + FFT + VGG + MS-SSIM   (main model)

Fairness is the whole point of an ablation, so the runner pins the things that
would otherwise drift between runs: same seed, same scene-disjoint split, same
epoch budget, same data. It refuses to start if the split leaks, and prints the
pinned settings so they can be quoted in a report.

Usage
-----
    # everything, defaults from config.yaml
    python run_ablations.py --config config.yaml

    # long run, detached (survives SSH disconnect)
    nohup python run_ablations.py --config config.yaml --epochs 150 \
          --batch-size 16 > ablations.log 2>&1 &
    tail -f ablations.log

    # just the two cheap rungs, shorter budget
    python run_ablations.py --ablations l1_only,l1_adv --epochs 50

Resuming
--------
Safe to re-run. Each config auto-resumes from its own latest checkpoint, and
configs that already produced metrics are skipped unless --force is passed.
This matters on a shared GPU where a run can be interrupted at any time.

Outputs
-------
    runs/{ablation}/config.yaml           exact config used
    checkpoints/{ablation}/best.pth       EMA weights, best val loss
    outputs/metrics_{ablation}_test.csv   per-config metrics
    outputs/ablation_comparison.csv       the table
    outputs/ablation_comparison.md        the same, paste-ready for the README
    outputs/ablation_comparison.png       bar chart
"""

import argparse
import copy
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path

import yaml

# Relative cost of one epoch per config, measured against `full` = 1.0.
# l1_only skips the discriminator and the VGG/FFT/MS-SSIM terms entirely, so it
# is far cheaper. Used only to print a time estimate up front.
_REL_COST = {"l1_only": 0.40, "l1_adv": 0.70, "l1_adv_fft": 0.75, "full": 1.00}

_ORDER = ["full", "l1_adv_fft", "l1_adv", "l1_only"]

_LABEL = {
    "l1_only":    "A: L1 only",
    "l1_adv":     "B: L1 + Adv",
    "l1_adv_fft": "C: L1 + Adv + FFT",
    "full":       "D: Full (ours)",
}


# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------

def check_gpu(allow_cpu: bool = False) -> str:
    import torch
    if not torch.cuda.is_available():
        if not allow_cpu:
            raise RuntimeError(
                "No CUDA device. An ablation study on CPU would take weeks — "
                "run this on the A5000, or pass --allow-cpu with a tiny "
                "--epochs/--subset-size to smoke-test the pipeline."
            )
        print("GPU  : none — running on CPU (--allow-cpu)")
        return "cpu"
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU  : {name}")
    print(f"VRAM : {vram:.1f} GB")
    return name


def audit_split(cfg: dict) -> None:
    """
    Verify no source scene spans two splits, and that every ablation will
    therefore be scored on the same untainted test set.
    """
    from data.dataloader import SARtoEODataset, _scene_key

    print("\n" + "=" * 66)
    print(" LEAKAGE AUDIT — shared split for all ablations")
    print("=" * 66)

    scenes = {}
    for split in ("train", "val", "test"):
        ds = SARtoEODataset(cfg, split=split, augment=False)
        scenes[split] = {_scene_key(p[0]) for p in ds.pairs}
        print(f"  {split:<6}: {len(ds.pairs):>7,} patches from "
              f"{len(scenes[split]):>5,} scenes")

    leaked = False
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = scenes[a] & scenes[b]
        print(f"  {a:<6} vs {b:<5}: "
              f"{'OK' if not shared else f'LEAK - {len(shared)} shared scenes'}")
        leaked |= bool(shared)

    if leaked:
        raise RuntimeError(
            "Split is leaking — test scenes also appear in train. Refusing to "
            "run: every number in the comparison table would be inflated, and "
            "inflated by a different amount per config, so the ranking would "
            "be meaningless too. Set split_strategy: 'scene' in the config."
        )

    n_scenes = len(scenes["train"]) + len(scenes["val"]) + len(scenes["test"])
    if n_scenes < 20:
        print(f"\n  NOTE: only {n_scenes} scene groups total. If the dataset is "
              f"large, _scene_key probably could not parse a scene id from the "
              f"filenames and fell back to directory grouping — which makes "
              f"this closer to a terrain split. Check a few filenames before "
              f"committing GPU time.")
    print("  PASSED — splits are scene-disjoint")
    print("=" * 66)


# ---------------------------------------------------------------------------
# One ablation
# ---------------------------------------------------------------------------

def run_one(base_cfg: dict, ablation: str, args) -> dict:
    """Train and evaluate a single config. Returns its metrics dict."""
    from train import train, make_dirs
    from eval import run_inference_to_dir, evaluate_dirs

    cfg = copy.deepcopy(base_cfg)
    cfg["active_ablation"] = ablation

    out_dir  = cfg["paths"]["output_dir"]
    run_dir  = os.path.join("runs", ablation)
    os.makedirs(run_dir, exist_ok=True)
    cfg_path = os.path.join(run_dir, "config.yaml")
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False,
                  allow_unicode=True)

    metrics_csv = os.path.join(out_dir, f"metrics_{ablation}_test.csv")
    if os.path.exists(metrics_csv) and not args.force:
        with open(metrics_csv, encoding="utf-8") as f:
            row = next(csv.DictReader(f))
        print(f"  Already evaluated — skipping (use --force to redo)")
        return {k: float(row[k]) for k in ("lpips", "fid", "ssim", "psnr")}

    make_dirs(cfg)
    t0 = time.time()
    train(cfg)                      # auto-resumes from its own checkpoints
    train_min = (time.time() - t0) / 60

    ckpt_dir = os.path.join(cfg["paths"]["checkpoint_dir"], ablation)
    weights  = os.path.join(ckpt_dir, "best.pth")
    if not os.path.exists(weights):
        # No validation ran (val_freq > epochs), so best.pth was never written.
        weights = os.path.join(ckpt_dir, "final.pth")
    if not os.path.exists(weights):
        raise FileNotFoundError(f"No checkpoint produced for '{ablation}'")

    pred_dir = os.path.join(out_dir, f"eval_preds_{ablation}")
    gt_dir   = os.path.join(out_dir, f"eval_gt_{ablation}")
    run_inference_to_dir(cfg_path, weights, "test", pred_dir, gt_dir,
                         use_tta=False)
    metrics = evaluate_dirs(pred_dir, gt_dir, metrics_csv, split="test")
    metrics["train_minutes"] = train_min
    return metrics


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_report(results: dict, out_dir: str, pinned: dict) -> None:
    """Emit the comparison table as CSV, markdown and a bar chart."""
    order = [a for a in ["l1_only", "l1_adv", "l1_adv_fft", "full"] if a in results]
    if not order:
        print("\nNo results to report.")
        return

    # Lower is better for lpips/fid, higher for ssim/psnr. Non-finite values
    # (FID is nan when pytorch-fid is missing) must not win "best".
    import math
    best = {}
    for k, lower_better in (("lpips", True), ("fid", True),
                            ("ssim", False), ("psnr", False)):
        vals = {a: results[a][k] for a in order
                if k in results[a] and math.isfinite(results[a][k])}
        if vals:
            best[k] = (min if lower_better else max)(vals, key=vals.get)

    csv_path = os.path.join(out_dir, "ablation_comparison.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["config", "label", "lpips", "fid", "ssim", "psnr",
                    "train_minutes"])
        for a in order:
            m = results[a]
            w.writerow([a, _LABEL[a], m.get("lpips", ""), m.get("fid", ""),
                        m.get("ssim", ""), m.get("psnr", ""),
                        round(m.get("train_minutes", 0), 1)])

    md_path = os.path.join(out_dir, "ablation_comparison.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| Configuration | Loss Components | LPIPS ↓ | FID ↓ | SSIM ↑ | PSNR ↑ |\n")
        f.write("|---|---|---|---|---|---|\n")
        comp = {
            "l1_only":    "L1",
            "l1_adv":     "L1 + Adv",
            "l1_adv_fft": "L1 + Adv + FFT",
            "full":       "L1 + Adv + FFT + VGG + MS-SSIM",
        }
        for a in order:
            m = results[a]
            cells = []
            for k, fmt in (("lpips", "{:.4f}"), ("fid", "{:.1f}"),
                           ("ssim", "{:.4f}"), ("psnr", "{:.2f} dB")):
                v = (fmt.format(m[k])
                     if k in m and math.isfinite(m[k]) else "—")
                cells.append(f"**{v}**" if best.get(k) == a else v)
            f.write(f"| {_LABEL[a]} | {comp[a]} | " + " | ".join(cells) + " |\n")
        ep = pinned["epochs"]
        f.write(f"\n*Scene-disjoint test split, {pinned['n_test']:,} patches. "
                f"All configs: {ep} epoch{'s' if ep != 1 else ''}, "
                f"batch {pinned['batch_size']}, seed {pinned['seed']}, "
                f"identical split. Bold = best.*\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        keys = [("ssim", "SSIM ↑"), ("psnr", "PSNR ↑ (dB)"),
                ("lpips", "LPIPS ↓"), ("fid", "FID ↓")]
        fig, axes = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 4.2))
        labels = [_LABEL[a].split(":")[0] for a in order]
        for ax, (k, title) in zip(axes, keys):
            vals = [results[a].get(k, 0) for a in order]
            bars = ax.bar(labels, vals,
                          color=["#4C78A8" if best.get(k) != a else "#F58518"
                                 for a in order])
            ax.set_title(title)
            ax.grid(axis="y", alpha=0.3)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                        f"{v:.3f}" if k in ("ssim", "lpips") else f"{v:.1f}",
                        ha="center", va="bottom", fontsize=9)
        fig.suptitle("Ablation study — scene-disjoint test split")
        fig.tight_layout()
        png = os.path.join(out_dir, "ablation_comparison.png")
        fig.savefig(png, dpi=140)
        plt.close(fig)
        print(f"  chart    -> {png}")
    except Exception as e:
        print(f"  (chart skipped: {e})")

    print(f"  csv      -> {csv_path}")
    print(f"  markdown -> {md_path}")

    print("\n" + "=" * 78)
    print(" ABLATION RESULTS — scene-disjoint test split")
    print("=" * 78)
    print(f"{'Configuration':<22}{'LPIPS':>10}{'FID':>10}{'SSIM':>10}"
          f"{'PSNR':>10}{'train':>10}")
    print("-" * 78)
    for a in order:
        m = results[a]
        cells = []
        for key, fmt in (("lpips", "{:.4f}"), ("fid", "{:.1f}"),
                         ("ssim", "{:.4f}"), ("psnr", "{:.2f}")):
            text = (fmt.format(m[key])
                    if key in m and math.isfinite(m[key]) else "—")
            cells.append(text + "*" if best.get(key) == a else text)
        print(f"{_LABEL[a]:<22}" + "".join(f"{c:>10}" for c in cells)
              + f"{m.get('train_minutes', 0) / 60:>9.1f}h")
    print("-" * 78)
    print(" * = best for that metric")
    print("=" * 78)

    # Two configs with genuinely different loss stacks should not land on the
    # same metrics to 4 decimals. When they do it almost always means the same
    # weights got evaluated twice, or eval read the wrong directory — the exact
    # fault that made the 2025 report's val and test columns identical. Cheap
    # to check, and silent failure here invalidates the whole table.
    seen = {}
    for a in order:
        sig = tuple(round(results[a][k], 4)
                    for k in ("lpips", "ssim", "psnr") if k in results[a])
        seen.setdefault(sig, []).append(a)
    clashes = [v for v in seen.values() if len(v) > 1]
    if clashes:
        print("\n  WARNING: identical metrics across different loss configs:")
        for group in clashes:
            print(f"    {' == '.join(group)}")
        print("  Different loss stacks producing identical scores usually means")
        print("  the same checkpoint was evaluated more than once. Verify that")
        print("  checkpoints/<config>/best.pth differ, and that each config's")
        print("  losses_<config>.csv shows only its own loss terms as non-zero.")
        print("  (Expected only if training was too short to diverge at all.)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Run the 4-config ablation study")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--ablations", default=",".join(_ORDER),
                   help="comma-separated; default runs the main model first so "
                        "an interrupted study still yields the headline result")
    p.add_argument("--epochs", type=int, default=None,
                   help="override epochs for ALL configs (fairness)")
    # Underscore spellings accepted too, matching the older scripts in this repo.
    p.add_argument("--batch-size", "--batch_size", type=int, default=None,
                   help="override batch size; A5000 24GB handles 16")
    p.add_argument("--subset-size", "--subset_size", type=int, default=None,
                   help="train on N pairs per split — for quick pilot runs")
    p.add_argument("--num-workers", "--num_workers", type=int, default=None,
                   help="DataLoader workers. The config default (4) is sized "
                        "for Kaggle's 2-core boxes; on a workstation use about "
                        "half your core count or the GPU will wait on PNG decode")
    p.add_argument("--force", action="store_true",
                   help="re-run configs that already have metrics")
    p.add_argument("--skip-audit", "--skip_audit", action="store_true",
                   help="skip the leakage audit (not recommended)")
    p.add_argument("--allow-cpu", "--allow_cpu", action="store_true",
                   help="permit running without a GPU, for pipeline smoke tests")
    args = p.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # ---- Pin the things that must not vary between configs ---------------
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.subset_size is not None:
        cfg["data"]["subset_size"] = args.subset_size
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = args.num_workers
    # A per-session cap makes sense on Kaggle; here it would silently truncate
    # some configs and not others, destroying comparability.
    cfg["training"].pop("session_epoch_limit", None)

    ablations = [a.strip() for a in args.ablations.split(",") if a.strip()]
    unknown = [a for a in ablations if a not in _LABEL]
    if unknown:
        p.error(f"unknown ablation(s): {unknown}. Choose from {list(_LABEL)}")

    check_gpu(allow_cpu=args.allow_cpu)

    epochs = cfg["training"]["epochs"]
    print(f"\nPinned across all configs:")
    print(f"  epochs         : {epochs}")
    print(f"  batch_size     : {cfg['training']['batch_size']}")
    print(f"  seed           : {cfg['training'].get('seed', 42)}")
    print(f"  split_strategy : {cfg['data'].get('split_strategy')}")
    print(f"  dataset_type   : {cfg['data'].get('dataset_type')}")
    print(f"  subset_size    : {cfg['data'].get('subset_size')}")
    print(f"  num_workers    : {cfg['data'].get('num_workers')}")
    print(f"\nQueue: {' -> '.join(ablations)}")

    if not args.skip_audit:
        audit_split(cfg)

    from data.dataloader import SARtoEODataset
    n_test = len(SARtoEODataset(cfg, split="test", augment=False).pairs)

    # Rough wall-clock estimate. ~2.5 min/epoch for `full` at batch 16 on an
    # A5000; scaled per config by _REL_COST. Real numbers replace this in the
    # final table.
    est = sum(_REL_COST.get(a, 1.0) for a in ablations) * epochs * 2.5 / 60
    print(f"\nEstimated total: ~{est:.1f} h (A5000, batch 16). "
          f"Actual times reported at the end.")

    out_dir = cfg["paths"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    pinned_meta = {"epochs": epochs,
                   "batch_size": cfg["training"]["batch_size"],
                   "seed": cfg["training"].get("seed", 42),
                   "n_test": n_test}

    results, failed = {}, {}
    for i, ablation in enumerate(ablations, 1):
        print("\n" + "#" * 78)
        print(f"# [{i}/{len(ablations)}]  {_LABEL[ablation]}   ({ablation})")
        print("#" * 78)
        try:
            results[ablation] = run_one(cfg, ablation, args)
            # Rewrite the comparison after every config, not only at the end.
            # This study runs across several sessions over days, and a table
            # that only appears once all four have finished is no use at a
            # review held partway through — nor if the machine is killed
            # rather than interrupted cleanly.
            write_report(results, out_dir, pinned_meta)
        except KeyboardInterrupt:
            print("\nInterrupted. Progress is checkpointed — re-run to resume.")
            break
        except Exception as e:
            # One config failing (OOM, say) should not discard the others.
            failed[ablation] = str(e)
            print(f"\n[FAILED] {ablation}: {e}")
            traceback.print_exc()
            print("Continuing with the remaining configs.\n")

    if results:
        write_report(results, out_dir, pinned_meta)

    if failed:
        print("\nFAILED CONFIGS:")
        for a, e in failed.items():
            print(f"  {a}: {e}")
        print("Re-run this script to retry them; finished configs are skipped.")

    with open(os.path.join(out_dir, "ablation_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump({"results": results, "failed": failed}, f, indent=2)


if __name__ == "__main__":
    # Windows consoles default to cp1252 and raise on the unicode used above.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    sys.path.insert(0, str(Path(__file__).parent))
    main()
