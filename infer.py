"""
infer.py — SAR-to-EO Inference Script

Input:  Directory of single-channel Sentinel-1 SAR (VV) images.
        8-bit PNG, dB-scaled and normalised to [0, 255]. Exactly 256×256
        takes the fast batch path; any other size is tiled automatically.

Output: Directory of generated RGB PNG images, same size and filenames as inputs.

Features:
  - Loads EMA model weights from checkpoint (best.pth)
  - Optional Test-Time Augmentation (--tta): average 4 rotation predictions
    for ~0.01–0.02 SSIM improvement at 4× inference time cost
  - Strided sliding-window tiling for scenes larger than one 256px tile
  - Batched inference with progress logging

Usage:
    python infer.py --input_dir <path> --output_dir <path> --weights <path>

    # With TTA (better quality, 4× slower):
    python infer.py --input_dir <path> --output_dir <path> --weights <path> --tta

Optional:
    --model_config config.yaml  (default: config.yaml in same directory)
    --device       cuda         (default: auto-detect)
    --batch_size   8            (default: 8, reduce if OOM)
    --tile         256          (sliding-window size; match training crop)
    --stride       192          (step between windows; smaller = smoother)
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
import torch.nn.functional as F


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
    full_skip = True

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
        full_skip  = m.get("full_res_skip",         True)

    G = UNetGenerator(
        in_channels  = in_ch,
        out_channels = out_ch,
        base_ch      = base_ch,
        use_attention= use_attn,
        pretrained   = False,       # weights come from checkpoint, not ImageNet
        gradient_checkpointing = grad_ck,
        full_res_skip = full_skip,
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

def load_sar_image(path: str, strict: bool = True) -> torch.Tensor:
    """
    Load a SAR patch conforming to the I/O contract:
      - 8-bit PNG, single-channel (grayscale)
      - dB-scaled and min-max normalised to [0, 255]
      - Exactly 256×256 pixels when strict=True

    strict=False is used by the tiling path, which accepts any size.

    Returns: [1, H, W] float32 tensor, normalised to [-1, 1]

    Raises:
        ValueError: if strict and dimensions are not exactly 256×256
    """
    img = Image.open(path).convert("L")

    if strict and img.size != (256, 256):
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
# Strided sliding-window inference for scenes larger than the 256px tile
# ---------------------------------------------------------------------------

def _blend_window(tile: int, device, dtype) -> torch.Tensor:
    """
    Separable raised-cosine (Hann) window, [1, 1, tile, tile].

    Weights each tile's contribution so it fades to ~0 at the edges. Averaging
    overlapping tiles with uniform weights still leaves visible seams, because
    a pixel at a tile boundary was predicted with almost no right-hand context
    while its neighbour had plenty. Tapering makes each pixel come mostly from
    the tile that saw the most context around it.

    Clamped away from exact zero so the accumulator can never divide by 0 in a
    corner covered by exactly one tile.
    """
    w = torch.hann_window(tile, periodic=False, device=device, dtype=dtype)
    w = w.clamp_min(1e-3)
    return (w[:, None] * w[None, :])[None, None]


def predict_large(G: torch.nn.Module,
                  sar: torch.Tensor,
                  tile: int = 256,
                  stride: int = 192,
                  use_amp: bool = True,
                  use_tta: bool = False,
                  batch_size: int = 8) -> torch.Tensor:
    """
    Run the generator over a SAR image of any size by striding a window across
    it and blending the overlaps.

    The model is fully convolutional but was trained only on 256×256 crops, so
    feeding it a whole scene at once drifts off-distribution (and blows up VRAM
    quadratically). Tiling keeps every forward pass at the trained size.

    `stride < tile` is what makes this work: neighbouring tiles overlap by
    `tile - stride` pixels, and the raised-cosine window blends them so no seam
    appears. stride=192 on a 256 tile gives 25% overlap, which is enough to hide
    boundaries without much extra compute. A smaller stride is smoother but
    costs (tile/stride)² forward passes.

    Args:
        G:          generator in eval mode
        sar:        [1, H, W] or [1, 1, H, W], range [-1, 1], any H/W
        tile:       window size — must match the training crop size
        stride:     step between windows; must be <= tile
        use_tta:    4-rotation TTA per tile (4× slower)
        batch_size: tiles per forward pass

    Returns:
        [3, H, W] prediction in [-1, 1], same spatial size as the input.
    """
    if stride > tile:
        raise ValueError(f"stride ({stride}) must be <= tile ({tile})")

    if sar.dim() == 3:
        sar = sar.unsqueeze(0)
    _, _, H, W = sar.shape
    device = sar.device

    # Reflect-pad so the window always lands fully inside the image, and so the
    # last row/column of tiles is not skipped when the size is not a multiple
    # of the stride. Reflection avoids inventing black borders the model would
    # then try to render.
    pad_h = max(0, tile - H) + (-(H - tile) % stride if H > tile else 0)
    pad_w = max(0, tile - W) + (-(W - tile) % stride if W > tile else 0)
    if pad_h or pad_w:
        # Reflect looks best, but it cannot pad by more than the dimension it
        # is mirroring — which happens whenever the scene is smaller than one
        # tile. Replicate has no such limit, so fall back to it in that case.
        mode = "reflect" if (pad_h < H and pad_w < W) else "replicate"
        sar = F.pad(sar, (0, pad_w, 0, pad_h), mode=mode)
    _, _, Hp, Wp = sar.shape

    ys = list(range(0, Hp - tile + 1, stride))
    xs = list(range(0, Wp - tile + 1, stride))

    acc    = torch.zeros((1, 3, Hp, Wp), device=device, dtype=torch.float32)
    weight = torch.zeros((1, 1, Hp, Wp), device=device, dtype=torch.float32)
    window = _blend_window(tile, device, torch.float32)

    coords = [(y, x) for y in ys for x in xs]
    for i in range(0, len(coords), batch_size):
        chunk = coords[i:i + batch_size]
        batch = torch.cat([sar[:, :, y:y + tile, x:x + tile] for y, x in chunk], 0)

        if use_tta:
            pred = tta_predict(G, batch, use_amp=use_amp)
        else:
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                    pred = G(batch)
        pred = pred.float()

        for j, (y, x) in enumerate(chunk):
            acc[:, :, y:y + tile, x:x + tile]    += pred[j:j + 1] * window
            weight[:, :, y:y + tile, x:x + tile] += window

    out = acc / weight
    return out[0, :, :H, :W]


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
    tile:         int  = 256,
    stride:       int  = 192,
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
    n_done    = 0
    n_total   = len(sar_files)   # fixed before the tiled files are split off

    # Anything that isn't exactly one tile goes through the strided sliding
    # window instead of the batch path, so a full Sentinel-1 scene works
    # without the caller having to pre-cut it.
    oversized = [f for f in sar_files if Image.open(f).size != (256, 256)]
    if oversized:
        print(f"[Infer] {len(oversized)} image(s) are not 256×256 — using "
              f"strided tiling (tile={tile}, stride={stride})")
        for f in oversized:
            sar = load_sar_image(str(f), strict=False).unsqueeze(0).to(device)
            pred = predict_large(G, sar, tile=tile, stride=stride,
                                 use_amp=use_amp, use_tta=use_tta,
                                 batch_size=batch_size)
            save_eo_image(pred, str(output_dir / f.name))
            n_done += 1
        sar_files = [f for f in sar_files if f not in set(oversized)]

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
            print(f"  {n_done}/{n_total} patches done...")

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
    parser.add_argument("--tile", type=int, default=256,
                        help="sliding-window size for images larger than one "
                             "tile; must match the training crop size")
    parser.add_argument("--stride", type=int, default=192,
                        help="step between windows (default 192 = 25%% overlap). "
                             "Smaller is smoother but costs (tile/stride)^2 passes")
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
        tile         = args.tile,
        stride       = args.stride,
    )
