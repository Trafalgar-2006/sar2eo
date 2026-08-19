"""
export_onnx.py — Export SAR2EO Generator to ONNX

ONNX export makes the model deployable outside of PyTorch:
  - Browser inference (ONNX Runtime Web)
  - Mobile deployment (CoreML, TFLite via ONNX)
  - C++ production inference
  - Shows production-readiness in portfolio

Usage:
    python export_onnx.py --weights checkpoints/full/best.pth --config config.yaml

Outputs:
    sar2eo_generator.onnx   (production model, ~135MB)
    sar2eo_generator_quant.onnx  (INT8 quantized, ~35MB)

Verify with:
    python export_onnx.py --verify
"""

import os
import sys
import yaml
import argparse
import numpy as np

import torch
import torch.nn as nn


def export_onnx(
    weights_path: str,
    config_path:  str,
    output_path:  str = "sar2eo_generator.onnx",
    opset:        int = 17,
    quantize:     bool = True,
):
    device = torch.device("cpu")  # ONNX export always on CPU

    # ── Load config and model ────────────────────────────────────────────
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    from models.generator import UNetGenerator
    m = cfg.get("model", {})
    G = UNetGenerator(
        in_channels=m.get("input_channels",  1),
        out_channels=m.get("output_channels", 3),
        base_ch=m.get("base_ch", 64),
        use_attention=m.get("use_attention", True),
        pretrained=False,
        full_res_skip=m.get("full_res_skip", True),
    ).to(device)

    ckpt  = torch.load(weights_path, map_location=device, weights_only=False)
    state = ckpt.get("G_ema") or ckpt.get("G") or ckpt
    G.load_state_dict(state)
    G.eval()

    meta = ckpt.get("meta", {})
    print(f"✓ Model loaded from: {weights_path}")
    if meta:
        print(f"  Trained at commit: {meta.get('git_commit', 'N/A')}")
        print(f"  PyTorch version:   {meta.get('torch_version', 'N/A')}")

    n_params = sum(p.numel() for p in G.parameters())
    print(f"  Parameters: {n_params/1e6:.1f}M")

    # ── Dummy input (single 256×256 SAR image) ───────────────────────────
    dummy = torch.randn(1, 1, 256, 256, device=device)

    print(f"\nExporting to ONNX (opset {opset}) ...")
    torch.onnx.export(
        G,
        dummy,
        output_path,
        opset_version=opset,
        input_names=["sar_input"],
        output_names=["eo_output"],
        dynamic_axes={
            "sar_input":  {0: "batch_size"},
            "eo_output":  {0: "batch_size"},
        },
        do_constant_folding=True,
        verbose=False,
    )
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"✓ ONNX model saved: {output_path} ({size_mb:.1f} MB)")

    # ── Verify ONNX output matches PyTorch ───────────────────────────────
    try:
        import onnxruntime as ort
        sess   = ort.InferenceSession(output_path)
        dummy_np = dummy.numpy()
        pt_out   = G(dummy).detach().numpy()
        ort_out  = sess.run(None, {"sar_input": dummy_np})[0]
        max_diff = np.abs(pt_out - ort_out).max()
        print(f"✓ ONNX verification: max diff PyTorch vs ONNX = {max_diff:.6f}")
        if max_diff > 1e-3:
            print("  ⚠ Diff > 1e-3 — check for non-deterministic ops")
    except ImportError:
        print("  (Install onnxruntime to verify: pip install onnxruntime)")

    # ── INT8 Quantization (optional) ─────────────────────────────────────
    if quantize:
        quant_path = output_path.replace(".onnx", "_quant_int8.onnx")
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            quantize_dynamic(output_path, quant_path, weight_type=QuantType.QInt8)
            size_q = os.path.getsize(quant_path) / 1e6
            print(f"✓ Quantized model: {quant_path} ({size_q:.1f} MB)")
            print(f"  Compression: {size_mb:.1f}MB → {size_q:.1f}MB "
                  f"({100*(1-size_q/size_mb):.0f}% smaller)")
        except ImportError:
            print("  (Install onnxruntime for quantization)")

    print("\n✓ ONNX export complete!")
    print(f"  Production model: {output_path}")
    if quantize:
        print(f"  Quantized model:  {quant_path}")
    print("\n  Inference example:")
    print("    import onnxruntime as ort, numpy as np")
    print(f"    sess = ort.InferenceSession('{output_path}')")
    print("    out  = sess.run(None, {'sar_input': sar_np})[0]")

    return output_path


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
    parser.add_argument("--weights", default="checkpoints/full/best.pth")
    parser.add_argument("--config",  default="config.yaml")
    parser.add_argument("--output",  default="sar2eo_generator.onnx")
    parser.add_argument("--opset",   type=int, default=17)
    parser.add_argument("--no-quantize", "--no_quantize", action="store_true")
    args = parser.parse_args()

    export_onnx(
        weights_path=args.weights,
        config_path=args.config,
        output_path=args.output,
        opset=args.opset,
        quantize=not args.no_quantize,
    )
