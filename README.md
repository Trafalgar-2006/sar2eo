# SAR → EO Image Translation
**Sentinel-1 SAR (VV) → Sentinel-2 RGB**

Personal research project. Phase 1 is also submitted as a 7th-semester Minor Project.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-ee4c2c?logo=pytorch)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Phase%201-implementation%20complete-blue)](#verification)

> Generate cloud-free Sentinel-2 optical imagery from Sentinel-1 SAR radar data using a conditional deep learning pipeline — no clouds, no waiting, any weather, any time.

---

## The Problem

Optical satellite imagery (Sentinel-2) is blocked by clouds ~67% of the time over most of Earth's surface. SAR (Synthetic Aperture Radar) penetrates clouds, rain, and darkness — but its grayscale backscatter images are hard for humans and downstream models to interpret.

This project learns to translate SAR → EO directly: given a Sentinel-1 patch, generate the corresponding Sentinel-2 RGB image that *would* exist if there were no clouds.

---

## Architecture

### Phase 1 — ResNet50-UNet + Multi-Scale GAN (Current)

Encoder and decoder are paired by spatial resolution — every skip joins two
stages of the same size.

```
resolution   ENCODER (ResNet50, ImageNet)        DECODER (from scratch)
  256 px     SAR in  [1,  256, 256] ───────────► d0 + out conv → [3, 256, 256]
  128 px     stem    [64,  128, 128] ──────────► d1   [64,  128, 128]
   64 px     layer1  [256,  64,  64] ──CBAM───► d2   [128,  64,  64]
   32 px     layer2  [512,  32,  32] ──CBAM───► d3   [256,  32,  32]
   16 px     layer3  [1024, 16,  16] ──CBAM───► d4   [512,  16,  16]
    8 px     layer4  [2048,  8,   8] ──CBAM───► bottleneck (proj4)
```

The 256 px skip is the one worth pointing at. ResNet50's stem halves the input
immediately, so **no encoder feature map exists at full resolution** — without
that connection the final decoder stage has to invent fine detail instead of
recovering it. It costs 9,568 parameters, 0.03% of the model. Set
`model.full_res_skip: false` to ablate it.

**Discriminator:** 3× PatchGAN judging at 256/128/64 px simultaneously — patch verdicts of 30×30, 14×14 and 6×6  
**Loss stack:** L1 (×100) + multi-scale GAN (×1) + FFT (×10) + VGG (×10) + MS-SSIM (×5)  
**Training:** EMA · cosine warmup · differential LR · gradient clipping · exact resume  
**Generator:** 35,537,419 params — 23,501,760 pretrained encoder + 12,035,659 decoder  
**Discriminator:** 8,299,971 params

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

## Evaluation protocol

**All splits are scene-disjoint.** This matters more than it sounds.

SEN1-2 does not contain independent images — it tiles each large scene on a
fixed **stride** grid, so patches with neighbouring indices overlap on the
ground. `ROIs1970_fall_s1_13_p265`, `p266` and `p267` are the same field shifted
one step at a time. Splitting on individual patches puts near-duplicates in both
train and test, and the model is then scored on pixels it memorised:

| Split strategy | Test scenes also seen in training |
|----------------|-----------------------------------|
| `random` (per-patch) | **100%** |
| `scene` (default) | **0%** |

`data/dataloader.py` groups patches by source scene before splitting, and
`python data/dataloader.py config.yaml` runs a leakage audit that fails loudly
if any scene spans two splits. The Kaggle notebook runs the same audit as a hard
gate before training starts.

Set via `split_strategy` in `config.yaml`:

- `scene` — 80/10/10 grouped by source scene (**default**)
- `terrain` — train on agri/barren/grassland, test on urban (hardest; measures cross-terrain generalisation)
- `random` — per-patch. Leaks. Retained only to reproduce pre-fix numbers.

---

## Results

| Model | SSIM ↑ | PSNR ↑ | LPIPS ↓ | FID ↓ |
|-------|--------|--------|---------|-------|
| Vanilla Pix2Pix (terrain split, 2025 baseline) | 0.073 | 12.2 dB | 0.615 | 278 |
| **ResNet50-UNet GAN (Phase 1)** — scene-disjoint | *training* | *training* | *training* | *training* |
| Conditional Diffusion (Phase 2) | *planned* | — | — | — |

*Results will be updated after training completes. Numbers are reported on a
scene-disjoint test split — see the evaluation protocol above. Figures from any
per-patch split are not comparable to these and are not reported.*

---

## Verification

The pipeline was reviewed defect-by-defect before any training time was spent.
**Seventeen defects were found and corrected.** Seven would each have
invalidated the reported results, and none of them raised an error at the time —
in every case the failure mode was plausible-looking output.

| Defect | Effect if left in |
|--------|-------------------|
| Train and test shared geographic ground | Every metric inflated; the model scored on pixels it had memorised |
| `init_weights()` re-randomised the ResNet50 encoder | `pretrained_encoder: true` silently did nothing; the lower encoder LR protected noise |
| EMA copied parameters but not buffers | Validation, `best.pth` and inference all ran on construction-time BatchNorm statistics |
| Nested archive layout matched only the first terrain | 4,000 of 16,000 pairs used, all agricultural — the exact cause of the earlier "cities rendered as farmland" failure |
| No full-resolution skip in the decoder | Finest detail interpolated rather than recovered; soft output |
| Resumed sessions replayed the previous session's augmentation | A 150-epoch run split three ways gave 50 distinct epochs repeated 3× |
| `terrain` split returned the same images for val and test | Checkpoints selected on the test set; the reported score is what it was tuned against |

Each fix is verified rather than asserted:

- **Split integrity** — test scenes also present in training fell from 100% to 0%,
  measured on the real 16,000-pair dataset; all pairs used exactly once.
- **Pretrained encoder** — `init_weights()` now alters encoder weights by
  `0.000e+00`, and they match torchvision `IMAGENET1K_V1` exactly afterwards.
- **Resume** — a run split 2+2 reproduces a continuous 4-epoch run with a maximum
  weight difference of `0.000e+00` across all 359 floating-point tensors.
- **Strided inference** — with an identity model, tiling and blending reconstruct
  the input to `2.4e-07` across eight image sizes; zero seam artefacts.
- **Augmentation** — SAR and EO stay aligned across all 16 flip/rotation
  combinations; SAR noise does not move geometry; EO brightness does not touch SAR.

Reproduce the split audit yourself:

```bash
python data/dataloader.py config.yaml
```

---

## Repository Structure

```
sar2eo/
├── models/
│   ├── generator.py           ResNet50-UNet + CBAM generator
│   ├── discriminator.py       Multi-scale (3×) PatchGAN
│   ├── losses.py              L1 · GAN · FFT · VGG · MS-SSIM
│   ├── attention.py           CBAM module
│   ├── diffusion/
│   │   ├── unet.py            Conditional denoising U-Net (40.4M)
│   │   └── ddpm.py            DDPM + DDIM sampler
│   └── controlnet/
│       └── controlnet.py      ControlNet adapter (Phase 3, scaffold)
├── data/
│   └── dataloader.py          Dataset · scene-disjoint split · leakage audit
├── utils/
│   ├── metrics.py             LPIPS · FID · SSIM · PSNR
│   ├── visualize.py           Loss curves · triplet grids
│   └── ema.py                 Exponential moving average
├── demo/
│   └── app.py                 Gradio demo (HuggingFace Spaces)
│
│   Phase 1
├── train.py                   Training loop · checkpointing · exact resume
├── run_ablations.py           4-config ablation study + comparison table
├── eval.py                    Evaluation on the held-out test split
├── eval_per_terrain.py        Per-terrain metric breakdown
├── infer.py                   Inference — batch · TTA · strided full-scene
├── plot_results.py            Loss curves and results figures
├── export_onnx.py             ONNX export (+ optional quantisation)
├── deploy_to_hf.py            HuggingFace Space deployment
│
│   Notebooks
├── local_train.ipynb          Local / lab GPU — 30 cells, step by step
├── kaggle_phase1_train.ipynb  Kaggle, with 12-hour session handling
├── kaggle_train.py            Kaggle single-cell variant
│
│   Phase 2 / 3
├── train_diffusion.py         Conditional DDPM
├── train_diffusion_ldm.py     Latent-diffusion variant
├── train_controlnet.py        ControlNet training (scaffold)
├── kaggle_train_diffusion.py  Kaggle diffusion runner
│
├── config.yaml                All hyperparameters, documented inline
└── requirements.txt           Pinned
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
# Phase 1 — single model
python train.py --config config.yaml

# Phase 2 — Diffusion (run after Phase 1)
python train_diffusion.py --config config.yaml
```

**On Kaggle:** import `kaggle_phase1_train.ipynb`. It discovers the dataset,
auto-resumes across the 12-hour session limit, and runs the leakage audit as a
hard gate before training starts.

### Ablation study

Trains all four loss configurations under pinned-identical conditions — same
seed, same scene-disjoint split, same epoch budget — then evaluates each on the
same held-out test set and writes the comparison table.

```bash
# all four, detached (survives SSH disconnect)
nohup python run_ablations.py --config config.yaml --epochs 150 \
      --batch-size 16 --num-workers 8 > ablations.log 2>&1 &
tail -f ablations.log

# quick pilot before committing GPU time
python run_ablations.py --epochs 5 --subset-size 200
```

| Flag | Purpose |
|------|---------|
| `--ablations` | Which configs, comma-separated. Default runs `full` first, so an interrupted study still yields the headline model |
| `--epochs` `--batch-size` `--subset-size` | Override for **all** configs — pinned so the comparison stays fair |
| `--num-workers` | DataLoader workers. The config default (4) is sized for Kaggle's 2-core boxes; on a workstation use about half your core count |
| `--force` | Re-run configs that already have metrics |
| `--allow-cpu` | Smoke-test the pipeline without booking GPU time |
| `--skip-audit` | Skip the leakage gate (not recommended) |

Flags accept either spelling — `--batch-size` and `--batch_size` both work,
across every script in the repo.

| Config | Loss components |
|--------|-----------------|
| `l1_only` | L1 |
| `l1_adv` | L1 + multi-scale adversarial |
| `l1_adv_fft` | + FFT frequency loss |
| `full` | + VGG perceptual + MS-SSIM (main model) |

Safe to re-run: each config auto-resumes from its own checkpoints and finished
configs are skipped. A config that fails (OOM, say) does not discard the others.
Outputs land in `outputs/ablation_comparison.{csv,md,png}` — the markdown is
paste-ready for the results table above.

---

## Inference

```bash
# Standard
python infer.py --input_dir ./sar_patches --output_dir ./eo_out --weights checkpoints/full/best.pth

# With Test-Time Augmentation (4× rotation ensemble, better quality)
python infer.py --input_dir ./sar_patches --output_dir ./eo_out --weights checkpoints/full/best.pth --tta
```

### Scenes larger than one tile

The model is trained on 256×256 crops, but SAR scenes aren't that size. Images
that aren't exactly one tile are handled automatically by striding a window
across them and blending the overlaps:

```bash
# a full scene — no pre-cutting needed
python infer.py --input-dir ./scenes --output-dir ./eo_out \
       --weights checkpoints/full/best.pth --stride 128
```

`stride` is the step between windows. The default of 192 on a 256 tile gives
25% overlap, blended with a raised-cosine window so no seam appears at the tile
boundaries — a pixel at the edge of one tile was predicted with little context
on that side, so it's weighted down in favour of the neighbouring tile that saw
more. A smaller stride is smoother but costs `(tile/stride)²` forward passes.

**I/O contract:** 8-bit grayscale PNG SAR → 8-bit RGB PNG EO, same dimensions
in and out. Exactly 256×256 takes the fast batch path; anything else is tiled.

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
