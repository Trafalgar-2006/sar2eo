"""
infer.py — SAR-to-EO Inference Script

Conforms exactly to the GalaxEye I/O contract:

  Input:  A directory of single-channel Sentinel-1 SAR (VV) patches,
          256×256 pixels, 8-bit PNG, dB-scaled and min–max normalised to [0, 255].

  Output: A directory of generated 256×256 RGB PNG images,
          same filenames as the corresponding inputs.

  CLI:    python infer.py --input_dir <path> --output_dir <path> --weights <path>

  Constraints:
    - Runs on a single GPU with ≤16 GB VRAM (Colab/Kaggle free tier)
    - No internet access at inference time (weights loaded locally)

Usage:
    python infer.py --input_dir /path/to/sar_patches \\
                    --output_dir /path/to/eo_output  \\
                    --weights checkpoints/full/best.pth

Optional:
    --model_config  config.yaml    (default: config.yaml in same directory)
    --device        cuda           (default: auto-detect)
    --batch_size    8              (default: 8, reduce if OOM)
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
# Default model parameters (used if config.yaml is unavailable)
# ---------------------------------------------------------------------------
DEFAULT_BASE_CHANNELS = 64
DEFAULT_IN_CHANNELS   = 1
DEFAULT_OUT_CHANNELS  = 3


def load_model(weights_path: str,
               config_path: str,
               device: torch.device) -> torch.nn.Module:
    """
    Load the UNet generator from a checkpoint.
    Falls back to default params if config.yaml is not found.
    """
    # Import here (after ensuring paths are correct)
    from models.generator import UNetGenerator

    # Try to load config
    base_ch  = DEFAULT_BASE_CHANNELS
    in_ch    = DEFAULT_IN_CHANNELS
    out_ch   = DEFAULT_OUT_CHANNELS

    if config_path and os.path.exists(config_path):
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        model_cfg = cfg.get("model", {})
        base_ch  = model_cfg.get("base_channels",  DEFAULT_BASE_CHANNELS)
        in_ch    = model_cfg.get("input_channels",  DEFAULT_IN_CHANNELS)
        out_ch   = model_cfg.get("output_channels", DEFAULT_OUT_CHANNELS)

    G = UNetGenerator(
        in_channels  = in_ch,
        out_channels = out_ch,
        base_ch      = base_ch,
    ).to(device)

    ckpt = torch.load(weights_path, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    if "G" in ckpt:
        G.load_state_dict(ckpt["G"])
    elif "state_dict" in ckpt:
        G.load_state_dict(ckpt["state_dict"])
    else:
        G.load_state_dict(ckpt)

    G.eval()
    return G


def load_sar_image(path: str) -> torch.Tensor:
    """
    Load a SAR patch conforming to the I/O contract:
      - 8-bit PNG, single-channel (grayscale)
      - dB-scaled and min–max normalised to [0, 255]

    Returns: [1, 256, 256] float32 tensor, normalised to [-1, 1]
    """
    img = Image.open(path).convert("L")   # Force grayscale

    # Verify dimensions
    if img.size != (256, 256):
        # Resize if necessary (should not happen with correctly prepared data)
        orig_size = img.size
        img = img.resize((256, 256), Image.BILINEAR)
        print(f"[WARNING] Resized {path} from {orig_size} to (256, 256)")

    arr = np.array(img, dtype=np.float32)   # [H, W], range [0, 255]
    arr = arr / 255.0                        # → [0, 1]
    arr = arr * 2.0 - 1.0                    # → [-1, 1]

    tensor = torch.from_numpy(arr).unsqueeze(0)   # [1, H, W]
    return tensor


def save_eo_image(tensor: torch.Tensor, path: str) -> None:
    """
    Save generated EO image from [-1, 1] tensor to 8-bit RGB PNG.
    Output: 256×256 RGB PNG, same filename as SAR input.
    """
    img = tensor.detach().cpu().float()
    img = (img + 1.0) / 2.0        # [-1, 1] → [0, 1]
    img = img.clamp(0.0, 1.0)
    img = img.permute(1, 2, 0)     # [3, H, W] → [H, W, 3]
    img = (img.numpy() * 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def run_inference(
    input_dir:    str,
    output_dir:   str,
    weights_path: str,
    config_path:  str  = "config.yaml",
    device_str:   str  = "auto",
    batch_size:   int  = 8,
) -> None:
    """
    Main inference function. Processes all PNG files in input_dir
    and writes RGB outputs to output_dir with matching filenames.
    """
    # ---- Device setup ----------------------------------------------------
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"[Infer] Device: {device}")

    # ---- Validate input directory ----------------------------------------
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
    print(f"[Infer] Found {len(sar_files)} SAR patches in {input_dir}")
    print(f"[Infer] Outputs -> {output_dir}")

    # ---- Load model -------------------------------------------------------
    print(f"[Infer] Loading model from {weights_path}...")
    G = load_model(weights_path, config_path, device)
    print(f"[Infer] Model loaded. Running inference...")

    # ---- Inference in batches --------------------------------------------
    use_amp = device.type == "cuda"
    n_processed = 0

    for i in range(0, len(sar_files), batch_size):
        batch_files = sar_files[i : i + batch_size]

        # Load batch
        batch_tensors = []
        for f in batch_files:
            t = load_sar_image(str(f))
            batch_tensors.append(t)

        batch = torch.stack(batch_tensors, dim=0).to(device)   # [B, 1, 256, 256]

        # Generate EO
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                fake_eo = G(batch)   # [B, 3, 256, 256]

        # Save each output with same filename as input
        for j, f in enumerate(batch_files):
            out_path = output_dir / f.name
            save_eo_image(fake_eo[j], str(out_path))
            n_processed += 1

        # Progress
        if (i // batch_size) % 10 == 0:
            print(f"  Processed {n_processed}/{len(sar_files)} patches...")

    print(f"\n[Infer] Done. Generated {n_processed} EO images -> {output_dir}")

    # ---- VRAM report (for reproducibility log) ---------------------------
    if device.type == "cuda":
        vram_used = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        print(f"[Infer] Peak VRAM used: {vram_used:.2f} GB")


# ---------------------------------------------------------------------------
# Entry point — must match the I/O contract exactly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SAR-to-EO inference (GalaxEye I/O contract)"
    )
    parser.add_argument(
        "--input_dir",  required=True,
        help="Directory of 256×256 8-bit PNG SAR (VV) patches"
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Directory to write generated 256×256 RGB EO PNGs"
    )
    parser.add_argument(
        "--weights",    required=True,
        help="Path to model checkpoint (.pth)"
    )
    parser.add_argument(
        "--model_config", default="config.yaml",
        help="Path to config.yaml (optional, uses defaults if not found)"
    )
    parser.add_argument(
        "--device",     default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to run inference on (default: auto)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Inference batch size (reduce if OOM, default: 8)"
    )

    args = parser.parse_args()

    run_inference(
        input_dir   = args.input_dir,
        output_dir  = args.output_dir,
        weights_path= args.weights,
        config_path = args.model_config,
        device_str  = args.device,
        batch_size  = args.batch_size,
    )
