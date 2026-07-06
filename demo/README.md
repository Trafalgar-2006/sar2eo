---
title: SAR to EO Image Translation
emoji: 🛰️
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
license: mit
short_description: Sentinel-1 SAR → Sentinel-2 RGB using ResNet50-UNet GAN
---

# 🛰️ SAR → EO Image Translation

**Sentinel-1 SAR (VV polarization) → Sentinel-2 RGB optical imagery**

Upload a 256×256 Sentinel-1 SAR patch and the model generates the corresponding
Sentinel-2 RGB image — cloud-free, any weather, any time of day.

## Model

- **Generator:** ResNet50-UNet + CBAM attention (35.5M params, pretrained ImageNet encoder)
- **Discriminator:** Multi-scale 3× PatchGAN (8.3M params)
- **Loss:** L1 + Adversarial + FFT frequency + VGG Perceptual + MS-SSIM
- **Training:** 150 epochs on Sentinel-1&2 Kaggle dataset (16K pairs)
- **EMA weights** used for inference (smoother outputs)

## How to Use

1. Upload a 256×256 SAR image (grayscale PNG)
2. Click Submit
3. Get the generated Sentinel-2 RGB image

## Source

[GitHub](https://github.com/Trafalgar-2006/sar2eo) · Built by [mohith-22000066](https://huggingface.co/mohith-22000066)
