"""
infer.py — SAR-to-EO Inference Script

Input:  Directory of single-channel Sentinel-1 SAR (VV) patches.
        Must be 256×256 pixels, 8-bit PNG, dB-scaled and normalised to [0, 255].

Output: Directory of generated 256×256 RGB PNG images, same filenames as inputs.

Features:
  - Loads EMA model weights from checkpoint (best.pth)
  - Optional Test-Time Augmentation (--tta): average 4 rotation predictions
    for ~0.01–0.02 SSIM improvement at 4× inference time cost
  - Strict input validation (raises ValueError on wrong size)
  - Batched inference with progress logging

Usage:
    python infer.py --input_dir <path> --output_dir <path> --weights <path>

    # With TTA (better quality, 4× slower):
    python infer.py --input_dir <path> --output_dir <path> --weights <path> --tta

Optional:
    --model_config config.yaml  (default: config.yaml in same directory)
    --device       cuda         (default: auto-detect)
    --batch_size   8            (default: 8, reduce if OOM)
    --tta                       (flag: enable test-time augmentation)
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.amp


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(weights_path: str,
               config_path: str,
               device: torch.device) -> torch.nn.Module:
    """
    Load the UNetGenerator from a checkpoint.

    Prefers EMA weights (G_ema key) over live weights (G key), since EMA
    weights are smoother and consistently produce better inference quality.
    Falls back to live weights if EMA not found (older checkpoints).
    """
    from models.generator import UNetGenerator

    # Parse config
    in_ch    = 1
    out_ch   = 3
    base_ch  = 64
    use_attn = True
    pretrained = True   # architecture flag (weights loaded from checkpoint)
    grad_ck  = False

    if config_path and os.path.exists(config_path):
        import yaml
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        m = cfg.get("model", {})
        in_ch      = m.get("input_channels",       1)
        out_ch     = m.get("output_channels",       3)
        base_ch    = m.get("base_ch",              64)
        use_attn   = m.get("use_attention",       True)
        grad_ck    = m.get("gradient_checkpointing", False)

    G = UNetGenerator(
        in_channels  = in_ch,
        out_channels = out_ch,
        base_ch      = base_ch,
        use_attention= use_attn,
        pretrained   = False,       # weights come from checkpoint, not ImageNet
        gradient_checkpointing = grad_ck,
    ).to(device)

    ckpt = torch.load(weights_path, map_location=device, weights_only=False)

    # Prefer EMA weights for inference
    if "G_ema" in ckpt:
        G.load_state_dict(ckpt["G_ema"])
        print(f"[Infer] Loaded EMA weights from checkpoint")
    elif "G" in ckpt:
        G.load_state_dict(ckpt["G"])
        print(f"[Infer] Loaded live weights from checkpoint (no EMA found)")
    elif "state_dict" in ckpt:
        G.load_state_dict(ckpt["state_dict"])
    else:
        G.load_state_dict(ckpt)

    G.eval()
    return G


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def load_sar_image(path: str) -> torch.Tensor:
    """
    Load a SAR patch conforming to the I/O contract:
      - 8-bit PNG, single-channel (grayscale)
      - dB-scaled and min-max normalised to [0, 255]
      - Exactly 256×256 pixels

    Returns: [1, 256, 256] float32 tensor, normalised to [-1, 1]

    Raises:
        ValueError: if image dimensions are not exactly 256×256
    """
    img = Image.open(path).convert("L")

    if img.size != (256, 256):
        raise ValueError(
            f"Input image must be exactly 256×256 pixels, "
            f"but got {img.size[0]}×{img.size[1]} for: {path}\n"
            f"Pre-process your SAR patches to 256×256 before running infer.py."
        )

    arr    = np.array(img, dtype=np.float32) / 255.0   # [0, 1]
    arr    = arr * 2.0 - 1.0                            # [-1, 1]
    return torch.from_numpy(arr).unsqueeze(0)           # [1, H, W]


def save_eo_image(tensor: torch.Tensor, path: str) -> None:
    """Save generated EO image from [-1, 1] tensor to 8-bit RGB PNG."""
    img = tensor.detach().cpu().float()
    img = (img + 1.0) / 2.0
    img = img.clamp(0.0, 1.0)
    img = img.permute(1, 2, 0)
    img = (img.numpy() * 255).astype(np.uint8)
    Image.fromarray(img).save(path)


# ---------------------------------------------------------------------------
# Test-Time Augmentation
# ---------------------------------------------------------------------------

def tta_predict(G: torch.nn.Module, batch: torch.Tensor,
                use_amp: bool = True) -> torch.Tensor:
    """
    4-rotation TTA: run inference at 0°, 90°, 180°, 270°, then
    un-rotate and average all predictions.

    Provides ~0.01–0.02 SSIM improvement at 4× inference time cost.
    Only worth using when quality is more important than speed.

    Args:
        G:       Generator in eval mode
        batch:   [B, 1, H, W] SAR input
        use_amp: Enable fp16 autocast

    Returns:
        [B, 3, H, W] averaged prediction
    """
    preds = []
    for k in range(4):
        rotated = torch.rot90(batch, k, dims=[2, 3]) if k > 0 else batch
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                pred = G(rotated)
        # Un-rotate prediction to original orientation
        if k > 0:
            pred = torch.rot90(pred, 4 - k, dims=[2, 3])
        preds.append(pred)
    return torch.stack(preds).mean(dim=0)


# ---------------------------------------------------------------------------
# Main inference function
# ---------------------------------------------------------------------------

def run_inference(
    input_dir:    str,
    output_dir:   str,
    weights_path: str,
    config_path:  str  = "config.yaml",
    device_str:   str  = "auto",
    batch_size:   int  = 8,
    use_tta:      bool = False,
) -> None:
    """
    Process all PNG files in input_dir and write RGB outputs to output_dir.
    """
    # Device
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"[Infer] Device: {device}")
    if use_tta:
        print(f"[Infer] TTA enabled (4× rotations, ~4× slower)")

    # Validate input
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        print(f"[ERROR] Input directory not found: {input_dir}")
        sys.exit(1)

    sar_files = sorted(input_dir.glob("*.png"))
    if not sar_files:
        print(f"[ERROR] No PNG files found in: {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Infer] {len(sar_files)} SAR patches → {output_dir}")

    # Load model
    print(f"[Infer] Loading model from {weights_path}...")
    G = load_model(weights_path, config_path, device)
    print(f"[Infer] Model ready. Running inference...")

    use_amp   = device.type == "cuda"
    predict   = tta_predict if use_tta else None
    n_done    = 0

    for i in range(0, len(sar_files), batch_size):
        batch_files   = sar_files[i : i + batch_size]
        batch_tensors = [load_sar_image(str(f)) for f in batch_files]
        batch         = torch.stack(batch_tensors).to(device)   # [B, 1, 256, 256]

        if use_tta:
            fake_eo = tta_predict(G, batch, use_amp=use_amp)
        else:
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                    fake_eo = G(batch)

        for j, f in enumerate(batch_files):
            save_eo_image(fake_eo[j], str(output_dir / f.name))
            n_done += 1

        if (i // batch_size) % 10 == 0:
            print(f"  {n_done}/{len(sar_files)} patches done...")

    print(f"\n[Infer] Done. {n_done} EO images written → {output_dir}")

    if device.type == "cuda":
        vram = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        print(f"[Infer] Peak VRAM: {vram:.2f} GB")


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

    parser = argparse.ArgumentParser(description="SAR-to-EO Inference")
    # Both spellings accepted throughout: hyphens are the argparse convention,
    # underscores are what this repo shipped with. argparse derives `dest` from
    # the first option string, so args.input_dir etc. are unchanged.
    parser.add_argument("--input-dir", "--input_dir", required=True,
                        help="Directory of 256×256 8-bit PNG SAR patches")
    parser.add_argument("--output-dir", "--output_dir", required=True,
                        help="Output directory for generated RGB EO PNGs")
    parser.add_argument("--weights",      required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--model-config", "--model_config", default="config.yaml",
                        help="Path to config.yaml")
    parser.add_argument("--device",       default="auto",
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("--batch-size", "--batch_size", type=int, default=8)
    parser.add_argument("--tta",          action="store_true",
                        help="Enable test-time augmentation (4× rotations, better quality)")

    args = parser.parse_args()
    run_inference(
        input_dir    = args.input_dir,
        output_dir   = args.output_dir,
        weights_path = args.weights,
        config_path  = args.model_config,
        device_str   = args.device,
        batch_size   = args.batch_size,
        use_tta      = args.tta,
    )
