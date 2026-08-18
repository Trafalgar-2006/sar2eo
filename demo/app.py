"""
app.py — SAR-to-EO Gradio Demo
Deploy on HuggingFace Spaces: https://huggingface.co/spaces

What it does:
  - Upload a Sentinel-1 SAR (VV) patch
  - Model generates the corresponding Sentinel-2 EO (RGB) image
  - Shows SAR input, generated EO, and CBAM attention overlay
  - Supports single image + batch from example gallery
"""

import os
import sys
import numpy as np
from PIL import Image
import torch
import gradio as gr

# ── Path setup (works locally and on HuggingFace Spaces) ──────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

WEIGHTS_PATH = os.path.join(ROOT, "checkpoints", "best.pth")
CONFIG_PATH  = os.path.join(ROOT, "config.yaml")
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load model once at startup ─────────────────────────────────────────────
def _load_model():
    from models.generator import UNetGenerator
    import yaml

    in_ch = out_ch = base_ch = 1
    use_attn = True

    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        m = cfg.get("model", {})
        in_ch    = m.get("input_channels",  1)
        out_ch   = m.get("output_channels", 3)
        base_ch  = m.get("base_ch",        64)
        use_attn = m.get("use_attention", True)

    G = UNetGenerator(
        in_channels=in_ch, out_channels=out_ch,
        base_ch=base_ch, use_attention=use_attn,
        pretrained=False,
    ).to(DEVICE)

    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(
            f"Model weights not found at {WEIGHTS_PATH}.\n"
            f"Place best.pth from Kaggle training in: checkpoints/best.pth"
        )

    ckpt = torch.load(WEIGHTS_PATH, map_location=DEVICE, weights_only=False)
    state = ckpt.get("G_ema") or ckpt.get("G") or ckpt
    G.load_state_dict(state)
    G.eval()
    return G

print(f"[Demo] Loading model on {DEVICE}...")
try:
    MODEL = _load_model()
    print(f"[Demo] Model ready ✓")
    MODEL_LOADED = True
except FileNotFoundError as e:
    print(f"[Demo] WARNING: {e}")
    MODEL = None
    MODEL_LOADED = False


# ── Inference ──────────────────────────────────────────────────────────────
def run_inference(pil_image: Image.Image, use_tta: bool = False) -> tuple:
    """
    Run SAR → EO inference.
    Returns (generated_eo_pil, attention_overlay_pil)
    """
    if MODEL is None:
        raise gr.Error("Model weights not loaded. See setup instructions.")

    # Validate size
    if pil_image.size != (256, 256):
        pil_image = pil_image.resize((256, 256), Image.LANCZOS)

    img = pil_image.convert("L")
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr * 2.0 - 1.0
    sar_t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(DEVICE)  # [1,1,256,256]

    use_amp = DEVICE.type == "cuda"

    with torch.no_grad():
        if use_tta:
            preds = []
            for k in range(4):
                rot = torch.rot90(sar_t, k, dims=[2, 3]) if k > 0 else sar_t
                with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                    pred = MODEL(rot)
                if k > 0:
                    pred = torch.rot90(pred, 4 - k, dims=[2, 3])
                preds.append(pred)
            fake = torch.stack(preds).mean(0)
        else:
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                fake = MODEL(sar_t)

    # Convert EO tensor → PIL
    eo = fake[0].cpu().float()
    eo = (eo + 1.0) / 2.0
    eo = eo.clamp(0, 1)
    eo_np = (eo.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    eo_pil = Image.fromarray(eo_np)

    # Attention overlay (colormap on SAR magnitude)
    sar_vis = ((arr + 1) / 2 * 255).astype(np.uint8)
    sar_rgb  = np.stack([sar_vis] * 3, axis=-1)
    overlay  = Image.fromarray(sar_rgb)   # simple SAR grayscale as RGB

    return eo_pil, overlay


# ── Gradio interface function ──────────────────────────────────────────────
def predict(sar_image, use_tta):
    if sar_image is None:
        return None, None, "⚠️ Please upload a SAR image."

    pil = Image.fromarray(sar_image).convert("L") if isinstance(sar_image, np.ndarray) else sar_image

    try:
        eo_pil, overlay_pil = run_inference(pil, use_tta=use_tta)
        status = (
            f"✅ Done — {'TTA (4× rotations)' if use_tta else 'Standard inference'} "
            f"on {str(DEVICE).upper()}"
        )
        return eo_pil, overlay_pil, status
    except Exception as e:
        return None, None, f"❌ Error: {str(e)}"


# ── Example SAR images ─────────────────────────────────────────────────────
EXAMPLES_DIR = os.path.join(ROOT, "demo", "examples")
EXAMPLES = []
if os.path.exists(EXAMPLES_DIR):
    EXAMPLES = [
        [os.path.join(EXAMPLES_DIR, f), False]
        for f in sorted(os.listdir(EXAMPLES_DIR))
        if f.endswith(".png")
    ][:6]


# ── UI ─────────────────────────────────────────────────────────────────────
CSS = """
.gradio-container { max-width: 1100px; margin: auto; }
.title { text-align: center; font-size: 2rem; font-weight: 700;
         background: linear-gradient(135deg, #667eea, #764ba2);
         -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.subtitle { text-align: center; color: #6b7280; margin-top: -0.5rem; }
.status-box { border-radius: 8px; padding: 0.5rem 1rem; font-size: 0.9rem; }
footer { display: none !important; }
"""

with gr.Blocks(css=CSS, theme=gr.themes.Soft()) as demo:

    gr.HTML("""
        <div class="title">SAR → EO Image Translation</div>
        <div class="subtitle">
          Sentinel-1 SAR (VV) → Sentinel-2 RGB · ResNet50-UNet + CBAM · Multi-Scale GAN
        </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📡 Input — Sentinel-1 SAR")
            sar_input = gr.Image(
                label="Upload SAR Patch (256×256 8-bit PNG)",
                type="numpy",
                height=280,
                image_mode="L",
            )
            use_tta = gr.Checkbox(
                label="🔄 Test-Time Augmentation (better quality, 4× slower)",
                value=False,
            )
            run_btn = gr.Button("🛰️ Generate EO Image", variant="primary", size="lg")

        with gr.Column(scale=1):
            gr.Markdown("### 🌍 Output — Generated Sentinel-2 EO")
            eo_output = gr.Image(
                label="Generated RGB Image",
                type="pil",
                height=280,
            )

        with gr.Column(scale=1):
            gr.Markdown("### 🔍 SAR Input (Grayscale View)")
            overlay_output = gr.Image(
                label="SAR Greyscale Overlay",
                type="pil",
                height=280,
            )

    status_box = gr.Textbox(
        label="Status", interactive=False, elem_classes=["status-box"]
    )

    run_btn.click(
        fn=predict,
        inputs=[sar_input, use_tta],
        outputs=[eo_output, overlay_output, status_box],
    )

    if EXAMPLES:
        gr.Examples(
            examples=EXAMPLES,
            inputs=[sar_input, use_tta],
            outputs=[eo_output, overlay_output, status_box],
            fn=predict,
            cache_examples=True,
        )

    gr.Markdown("""
    ---
    ### About

    This model translates **Sentinel-1 SAR (VV polarization)** patches into
    **Sentinel-2 RGB optical** imagery using a conditional GAN with:

    - **Generator:** ResNet50-UNet with CBAM attention (35.5M params, pretrained encoder)
    - **Discriminator:** Multi-scale PatchGAN at 3 resolutions
    - **Loss stack:** L1 + Adversarial + FFT frequency + VGG perceptual + MS-SSIM
    - **Training:** EMA weights · Cosine warmup LR · Gradient clipping

    **Input:** 256×256 8-bit grayscale PNG, dB-scaled SAR (VV)  
    **Output:** 256×256 8-bit RGB PNG

    📦 [GitHub](https://github.com/Trafalgar-2006/sar2eo) ·
    🗃️ Dataset: [Kaggle Sentinel-1&2](https://www.kaggle.com/datasets/requiemonk/sentinel12-image-pairs-segregated-by-terrain)
    """)

if __name__ == "__main__":
    # Windows consoles default to cp1252 and raise on the unicode used in the
    # progress output below. Force UTF-8 so local runs match Kaggle/Linux.
    import sys as _sys
    try:
        _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    demo.launch(share=True, server_name="0.0.0.0")
