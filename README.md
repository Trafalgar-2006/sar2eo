# SAR → EO Image Translation
**Sentinel-1 SAR (VV) → Sentinel-2 RGB — Personal Research Project**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-ee4c2c?logo=pytorch)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/Demo-HuggingFace%20Spaces-yellow?logo=huggingface)](https://huggingface.co/spaces)

> Generate cloud-free Sentinel-2 optical imagery from Sentinel-1 SAR radar data using a conditional deep learning pipeline — no clouds, no waiting, any weather, any time.

---

## The Problem

Optical satellite imagery (Sentinel-2) is blocked by clouds ~67% of the time over most of Earth's surface. SAR (Synthetic Aperture Radar) penetrates clouds, rain, and darkness — but its grayscale backscatter images are hard for humans and downstream models to interpret.

This project learns to translate SAR → EO directly: given a Sentinel-1 patch, generate the corresponding Sentinel-2 RGB image that *would* exist if there were no clouds.

---

## Architecture

### Phase 1 — ResNet50-UNet + Multi-Scale GAN (Current)

```
SAR [1, 256, 256]
      ↓
ResNet50 Encoder (pretrained ImageNet)
  stem   → [64,  128, 128]   ┐
  layer1 → [256,  64,  64]   │ CBAM attention
  layer2 → [512,  32,  32]   │ on every skip
  layer3 → [1024, 16,  16]   │
  layer4 → [2048,  8,   8]   ┘
      ↓
Bilinear Upsample Decoder
      ↓
EO RGB [3, 256, 256]
```

**Discriminator:** 3× PatchGAN at 256px / 128px / 64px simultaneously  
**Loss stack:** L1 + Multi-scale GAN + FFT + VGG perceptual + MS-SSIM  
**Training:** EMA · Cosine warmup · Differential LR · Gradient clipping  
**Generator params:** 35.5M  |  **Discriminator params:** 8.3M

### Phase 2 — Conditional Diffusion Model (Next)

```
SAR → ResNet50 encoder → conditioning features
Pure noise → Conditional U-Net (40M params) → denoise 1000→0 steps
                     ↑ time embedding injected at every ResBlock
DDIM fast sampler: 50 steps (deterministic, high quality)
```

### Phase 3 — ControlNet / Foundation Model (Planned)

Fine-tune Clay or Prithvi geospatial foundation models with ControlNet-style SAR conditioning.

---

## Results

| Model | SSIM ↑ | PSNR ↑ | LPIPS ↓ | FID ↓ |
|-------|--------|--------|---------|-------|
| Vanilla Pix2Pix (baseline) | 0.073 | 12.2 dB | 0.615 | 278 |
| **ResNet50-UNet GAN (Phase 1)** | *training* | *training* | *training* | *training* |
| Conditional Diffusion (Phase 2) | *planned* | — | — | — |

*Results will be updated after training completes.*

---

## Repository Structure

```
sar2eo/
├── models/
│   ├── generator.py        ResNet50-UNet + CBAM generator
│   ├── discriminator.py    Multi-scale (3×) PatchGAN
│   ├── losses.py           L1 · GAN · FFT · VGG · MS-SSIM
│   ├── attention.py        CBAM module
│   └── diffusion/
│       ├── unet.py         Conditional denoising U-Net (40M)
│       └── ddpm.py         DDPM + DDIM scheduler
├── data/
│   └── dataloader.py       Combined SEN1-2 + Kaggle dataset loader
├── utils/
│   ├── metrics.py          LPIPS · FID · SSIM · PSNR
│   ├── visualize.py        Loss curves · triplet grids
│   └── ema.py              Exponential Moving Average
├── demo/
│   └── app.py              Gradio demo (HuggingFace Spaces)
├── train.py                GAN training (Phase 1)
├── train_diffusion.py      Diffusion training (Phase 2)
├── eval.py                 Evaluation script
├── infer.py                Inference (+ TTA option)
├── kaggle_train.py         Single-cell Kaggle notebook script
├── config.yaml             All hyperparameters
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/Trafalgar-2006/sar2eo.git
cd sar2eo
pip install -r requirements.txt
```

### Datasets

**Kaggle Sentinel-1&2** (16,000 pairs, 4 terrain classes):
```
kaggle datasets download requiemonk/sentinel12-image-pairs-segregated-by-terrain
```

**SEN1-2** (280,000 pairs, 4 seasons — TU Munich, CC-BY 4.0):
```bash
rsync -avz rsync://m1436631@dataserv.ub.tum.de/m1436631/ ./data/SEN1-2/
# Password: m1436631
```

Configure in `config.yaml`:
```yaml
data:
  dataset_type: "combined"   # "kaggle" | "sen12" | "combined"
```

---

## Training

```bash
# Phase 1 — GAN
python train.py --config config.yaml

# Phase 2 — Diffusion (run after Phase 1)
python train_diffusion.py --config config.yaml

# On Kaggle (single cell):
exec(open("kaggle_train.py").read())
```

---

## Inference

```bash
# Standard
python infer.py --input_dir ./sar_patches --output_dir ./eo_out --weights checkpoints/full/best.pth

# With Test-Time Augmentation (4× rotation ensemble, better quality)
python infer.py --input_dir ./sar_patches --output_dir ./eo_out --weights checkpoints/full/best.pth --tta
```

**I/O contract:** 256×256 8-bit grayscale PNG SAR → 256×256 8-bit RGB PNG EO

---

## Evaluation

```bash
python eval.py --config config.yaml --weights checkpoints/full/best.pth --split test
```

Metrics: LPIPS ↓ · FID ↓ · SSIM ↑ · PSNR ↑

---

## References

**Datasets**
- Schmitt, M. (2018). SEN1-2. TU Munich. https://doi.org/10.14459/2018mp1436631
- Tiwari, P. (2021). Sentinel-1&2 Image Pairs. Kaggle.

**Architecture**
- Isola et al. (2017). Pix2Pix. https://arxiv.org/abs/1611.07004
- Wang et al. (2018). Pix2PixHD. https://arxiv.org/abs/1711.11585
- Woo et al. (2018). CBAM. https://arxiv.org/abs/1807.06521
- Rombach et al. (2022). LDM / Stable Diffusion. https://arxiv.org/abs/2112.10752
- Song et al. (2020). DDIM. https://arxiv.org/abs/2010.02502
- Zhang & Agrawala (2023). ControlNet. https://arxiv.org/abs/2302.05543

---

## License

MIT License. See [LICENSE](LICENSE).
