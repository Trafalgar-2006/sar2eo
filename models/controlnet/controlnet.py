"""
models/controlnet/controlnet.py — ControlNet for SAR-conditioned SD

Wraps a frozen Stable Diffusion UNet with a trainable ControlNet adapter.
The ControlNet takes SAR as structural control; frozen SD provides the
natural image prior (textures, colours, photorealism).

Architecture:
  Frozen SD UNet (3.5B params) — NEVER updated
  ControlNet    (~860M params)  — trainable copy of SD encoder
        ↑ SAR image [1, 256, 256] → preprocessed → ControlNet
        → zero-conv residuals injected into SD UNet decoder

References:
  Zhang & Agrawala (2023). Adding Conditional Control to Text-to-Image
  Diffusion Models. https://arxiv.org/abs/2302.05543

Usage:
  pip install diffusers transformers accelerate

  python train_controlnet.py --config config.yaml
"""

import torch
import torch.nn as nn
from typing import Optional


class ZeroConv(nn.Module):
    """
    Zero-initialised 1×1 convolution.
    This is the core ControlNet trick — zero init means the adapter
    starts as a no-op and learns to inject signal gradually.
    Without this, random initial residuals would destroy the pretrained SD.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SARControlNetConditioner(nn.Module):
    """
    Lightweight SAR → control-signal preprocessor.

    Takes a 1-channel SAR image and projects it to 3-channel
    "hint" image suitable for the ControlNet encoder.

    Architecture:
        Conv 1→16 → BN → ReLU
        Conv 16→32 → BN → ReLU
        Conv 32→3             ← SD expects 3-channel hint

    This is much simpler than a full encoder — the SD ControlNet
    encoder does the heavy lifting.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 1),
            nn.Tanh(),   # output in [-1, 1] — same range as SD input
        )

    def forward(self, sar: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sar: [B, 1, H, W] SAR in [-1, 1]
        Returns:
            [B, 3, H, W] control hint for ControlNet
        """
        return self.net(sar)


# ---------------------------------------------------------------------------
# HuggingFace Diffusers integration
# ---------------------------------------------------------------------------

def build_controlnet_pipeline(
    base_model: str = "runwayml/stable-diffusion-v1-5",
    device: str = "cuda",
):
    """
    Build a ControlNet fine-tuning pipeline using HuggingFace diffusers.

    Returns:
        (unet, controlnet, vae, noise_scheduler, tokenizer, text_encoder)

    Usage in train_controlnet.py:
        pipeline = build_controlnet_pipeline()
        # Freeze SD UNet + VAE + text encoder
        # Only train: controlnet + sar_conditioner
    """
    try:
        from diffusers import (
            StableDiffusionControlNetPipeline,
            ControlNetModel,
            DDPMScheduler,
            AutoencoderKL,
            UNet2DConditionModel,
        )
        from transformers import CLIPTokenizer, CLIPTextModel
    except ImportError:
        raise ImportError(
            "Install HuggingFace diffusers:\n"
            "  pip install diffusers transformers accelerate"
        )

    print(f"Loading SD base model: {base_model}")
    print("  This downloads ~4GB on first run (cached after)")

    # Load components separately for fine-grained control
    vae          = AutoencoderKL.from_pretrained(base_model, subfolder="vae")
    unet         = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet")
    tokenizer    = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder")
    scheduler    = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler")
    controlnet   = ControlNetModel.from_unet(unet)   # copy of SD encoder

    # Freeze everything except ControlNet
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet.train()    # only this is trained

    # Move to device
    vae          = vae.to(device)
    unet         = unet.to(device)
    text_encoder = text_encoder.to(device)
    controlnet   = controlnet.to(device)

    n_trainable = sum(p.numel() for p in controlnet.parameters() if p.requires_grad)
    n_frozen    = (
        sum(p.numel() for p in vae.parameters()) +
        sum(p.numel() for p in unet.parameters()) +
        sum(p.numel() for p in text_encoder.parameters())
    )
    print(f"  Trainable (ControlNet): {n_trainable/1e6:.1f}M params")
    print(f"  Frozen (SD):            {n_frozen/1e6:.1f}M params")

    return dict(
        unet=unet, controlnet=controlnet, vae=vae,
        scheduler=scheduler, tokenizer=tokenizer, text_encoder=text_encoder,
    )


def encode_prompt(tokenizer, text_encoder, prompt: str, device: str) -> torch.Tensor:
    """Encode text prompt to CLIP embeddings (used as SD condition)."""
    tokens = tokenizer(
        prompt, padding="max_length", max_length=tokenizer.model_max_length,
        truncation=True, return_tensors="pt"
    ).input_ids.to(device)
    with torch.no_grad():
        return text_encoder(tokens)[0]


# ---------------------------------------------------------------------------
# Null-text conditioning (for SAR-only, no text prompt)
# ---------------------------------------------------------------------------

# We want SAR to fully control the generation, not text.
# Use a fixed neutral text prompt — the ControlNet does all the work.
NULL_PROMPT = "satellite optical image, high quality, clear"
