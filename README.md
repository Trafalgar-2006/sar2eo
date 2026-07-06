# SAR-to-EO Image Translation

> Generating Sentinel-2 optical imagery from Sentinel-1 SAR (VV) using a pretrained ResNet50-UNet with CBAM attention and multi-scale adversarial training.

---

## Overview

This project implements a high-quality SAR-to-EO image translation pipeline.
Given a single-channel Sentinel-1 SAR (VV) patch, the model generates a plausible
Sentinel-2 RGB optical image of the same scene.

**Architecture highlights:**
- **ResNet50-UNet generator** — pretrained ImageNet encoder (1-channel adapted) + CBAM skip attention + bilinear upsample decoder
- **Multi-scale PatchGAN discriminator** — 3 independent PatchGANs at 1×, 0.5×, 0.25× resolution
- **5-component loss stack** — L1 + FFT frequency + VGG perceptual + MS-SSIM + adversarial
- **EMA model weights** — shadow generator averages training noise → stable inference
- **Test-time augmentation** — optional 4-rotation ensemble for better inference quality

---

## Repository Structure

```
sar2eo/
├── models/
│   ├── generator.py      ResNet50-UNet + CBAM generator
│   ├── discriminator.py  Multi-scale (3×) PatchGAN discriminator
│   ├── losses.py         L1, GAN, FFT, VGG, MS-SSIM losses
│   └── attention.py      CBAM module (channel + spatial attention)
├── data/
│   └── dataloader.py     SEN1-2 + Kaggle combined dataset loader
├── utils/
│   ├── metrics.py        LPIPS, FID, SSIM, PSNR computation
│   ├── visualize.py      Loss curves + SAR/EO/GT triplet grids
│   └── ema.py            EMA wrapper for generator weights
├── train.py              Training loop (EMA, cosine warmup, grad clip)
├── eval.py               Evaluation (metrics + TTA option)
├── infer.py              Inference (strict I/O contract, TTA flag)
├── config.yaml           All hyperparameters
├── requirements.txt      Pinned dependencies
└── kaggle_train.ipynb    Kaggle training notebook
```

---

## Architecture

### Generator — ResNet50-UNet + CBAM

```
Input: [B, 1, 256, 256] SAR (VV)
                ↓
        ResNet50 Encoder (pretrained ImageNet, 1-ch adapted)
        ┌─────────────────────────────────────────┐
        │ stem   → [B, 64,  128, 128]             │
        │ layer1 → [B, 256,  64,  64]             │
        │ layer2 → [B, 512,  32,  32]             │
        │ layer3 → [B, 1024, 16,  16]             │
        │ layer4 → [B, 2048,  8,   8]  bottleneck │
        └─────────────────────────────────────────┘
                ↓ channel projections (reduce VRAM)
                ↓ CBAM on each skip (channel + spatial attention)
        ┌─────────────────────────────────────────┐
        │ Bilinear Upsample Decoder               │
        │ d4: 512  + 512  → 512   [8→16]          │
        │ d3: 512  + 256  → 256   [16→32]         │
        │ d2: 256  + 128  → 128   [32→64]         │
        │ d1: 128  +  64  →  64   [64→128]        │
        │ d0:  64  (no skip) → 32 [128→256]       │
        └─────────────────────────────────────────┘
                ↓ 1×1 conv + Tanh
Output: [B, 3, 256, 256] EO (RGB)
```

**Why bilinear upsample over ConvTranspose2d:**
ConvTranspose2d is known to produce checkerboard artefacts in generated images.
Bilinear upsample followed by regular convolutions avoids this entirely.

**Why CBAM on skip connections:**
Not all ResNet50 features are relevant to SAR. CBAM channel attention suppresses
irrelevant ImageNet filters (e.g. colour-sensitive detectors) and amplifies SAR-useful
ones. Spatial attention focuses on informative regions (boundaries, structures) vs.
diffuse speckle areas.

### Discriminator — Multi-Scale PatchGAN

3 independent 70×70 PatchGANs operating at different resolutions:
- `D_0`: 256×256 — fine texture discrimination
- `D_1`: 128×128 — medium-scale structure
- `D_2`: 64×64  — global layout / coherence

Loss averaged across scales. Forces the generator to be realistic at ALL scales
simultaneously, not just the one the discriminator is tuned to.

### Loss Stack

| Loss | Weight | What it targets |
|------|--------|----------------|
| L1 | 100 | Pixel accuracy, colour fidelity, PSNR |
| Adversarial (multi-scale) | 1 | Sharpness, texture realism, FID |
| FFT Magnitude | 10 | High-frequency content (SAR speckle physics) |
| VGG Perceptual | 10 | Semantic feature similarity, LPIPS |
| MS-SSIM | 5 | Structural similarity at multiple scales, SSIM |

---

## Requirements

- Python 3.10+
- CUDA GPU (≥8 GB VRAM for training; ≥4 GB for inference)
- Tested on Kaggle P100 (16 GB)

```bash
pip install -r requirements.txt
```

---

## Dataset

### SEN1-2 (TU Munich, CC-BY 4.0)

```bash
rsync -avz rsync://m1436631@dataserv.ub.tum.de/m1436631/ ./data/SEN1-2/
# Password: m1436631
```

```
data/SEN1-2/
├── ROIs1158_spring/
│   ├── s1_1/    ← SAR grayscale PNGs
│   └── s2_1/    ← EO RGB PNGs
├── ROIs1868_summer/
├── ROIs1970_fall/
└── ROIs2017_winter/
```

### Kaggle Sentinel-1&2 Terrain Dataset

Download from [Kaggle](https://www.kaggle.com/datasets/requiemonk/sentinel12-image-pairs-segregated-by-terrain):

```
data/sentinel12/
├── agri/        ├── s1/  └── s2/
├── barrenland/  ├── s1/  └── s2/
├── grassland/   ├── s1/  └── s2/
└── urban/       ├── s1/  └── s2/
```

### Combined mode (recommended)

Set `data.dataset_type: "combined"` in `config.yaml` to pool both datasets
and use a random 80/10/10 split. This gives the most diverse training set.

---

## Training

```bash
# Full model (recommended)
python train.py --config config.yaml

# Specific ablation
python train.py --config config.yaml --ablation full
python train.py --config config.yaml --ablation l1_only
```

**Kaggle training** — clone repo, mount datasets, run:
```python
!git clone https://github.com/Trafalgar-2006/sar2eo.git
%cd sar2eo
!pip install -r requirements.txt
!python train.py --config config.yaml
```

**Checkpoints** saved to `checkpoints/{ablation}/`:
- `best.pth` — best validation loss (EMA weights)
- `epoch_N.pth` — periodic saves every 10 epochs
- `final.pth` — last epoch

**Logs** in `logs/{ablation}_steps.jsonl` — per-step losses (every 50 steps) for smooth curve plotting.

---

## Inference

```bash
python infer.py \
  --input_dir  <path/to/sar_patches> \
  --output_dir <path/to/eo_output>   \
  --weights    checkpoints/full/best.pth

# With test-time augmentation (better quality, 4× slower):
python infer.py --input_dir <...> --output_dir <...> --weights <...> --tta
```

**I/O contract:**
- Input: 256×256 8-bit grayscale PNG, dB-scaled SAR (VV) patches
- Output: 256×256 8-bit RGB PNG EO images, same filenames

---

## Evaluation

```bash
# Auto-run inference + all metrics:
python eval.py \
  --config  config.yaml \
  --weights checkpoints/full/best.pth \
  --split   test

# With TTA:
python eval.py --config config.yaml --weights checkpoints/full/best.pth --split test --tta
```

Metrics computed:
| Metric | Direction | Notes |
|--------|-----------|-------|
| LPIPS | ↓ lower | Learned perceptual similarity (AlexNet features) |
| FID | ↓ lower | Fréchet Inception Distance (distribution-level) |
| SSIM | ↑ higher | Structural similarity |
| PSNR | ↑ higher | Peak signal-to-noise ratio |

---

## Results

*To be updated after retraining with combined dataset + full architecture.*

Expected improvements over baseline vanilla U-Net:

| Model | SSIM ↑ | PSNR ↑ | LPIPS ↓ |
|-------|--------|--------|---------|
| Vanilla Pix2Pix U-Net (baseline) | 0.073 | 12.2 dB | 0.615 |
| ResNet50-UNet + CBAM + Multi-scale D (this) | ~0.30–0.40 | ~16–18 dB | ~0.40–0.50 |

---

## References

**Datasets:**
```
Schmitt, M. (2018). SEN1-2. Technical University of Munich.
https://doi.org/10.14459/2018mp1436631. CC-BY 4.0.

Tiwari, P. (2021). Sentinel-1&2 Image Pairs (Kaggle).
https://www.kaggle.com/datasets/requiemonk/sentinel12-image-pairs-segregated-by-terrain
```

**Papers:**
```
Isola et al. (2017). Image-to-Image Translation with Conditional Adversarial Networks.
CVPR 2017. https://arxiv.org/abs/1611.07004

Wang et al. (2018). High-Resolution Image Synthesis with Conditional GANs (pix2pixHD).
CVPR 2018. https://arxiv.org/abs/1711.11585

Woo et al. (2018). CBAM: Convolutional Block Attention Module.
ECCV 2018. https://arxiv.org/abs/1807.06521

He et al. (2016). Deep Residual Learning for Image Recognition.
CVPR 2016. https://arxiv.org/abs/1512.03385

Wang et al. (2003). Multi-scale structural similarity for image quality assessment.
Asilomar 2003. https://doi.org/10.1109/ACSSC.2003.1292216

Zhang et al. (2018). The Unreasonable Effectiveness of Deep Features as a Perceptual Metric.
CVPR 2018. https://arxiv.org/abs/1801.03924
```
