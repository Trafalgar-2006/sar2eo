# SAR-to-EO Image Translation
**GalaxEye Technical Assignment — AI Research Intern**

> Given a Sentinel-1 SAR (VV) image, generate the corresponding Sentinel-2 optical (RGB) image.

---

## Repository File Map

| File | Purpose |
|---|---|
| `model.py` | Root-level shim — re-exports `UNetGenerator`, `PatchGANDiscriminator`, all losses |
| `dataloader.py` | Root-level shim — re-exports `SARtoEODataset`, `get_dataloaders` |
| `train.py` | Training loop with mixed precision, LR decay, checkpointing |
| `eval.py` | Evaluation — runs inference + computes LPIPS/FID/SSIM/PSNR |
| `infer.py` | Inference per GalaxEye I/O contract |
| `config.yaml` | All hyperparameters (LR, batch, epochs, losses, seed, augmentation) |
| `requirements.txt` | Pinned dependencies |
| `models/generator.py` | U-Net generator (8 encoder + 8 decoder + skip connections) |
| `models/discriminator.py` | 70×70 PatchGAN discriminator |
| `models/losses.py` | L1, GAN, FFT frequency, VGG perceptual losses |
| `data/dataloader.py` | Full dataset implementation (SEN1-2 + Kaggle terrain-split) |
| `utils/metrics.py` | LPIPS, FID, SSIM, PSNR computation |
| `utils/visualize.py` | Loss curve plots + SAR/EO/GT triplet grids |
| `kaggle_train.ipynb` | End-to-end training notebook for Kaggle T4 GPU |

---

## Approach

This project implements a **Pix2Pix-based conditional GAN** with a non-standard loss stack designed to directly target the primary evaluation metrics (LPIPS ↓, FID ↓):

| Loss Component | Motivation |
|---|---|
| **L1** | Pixel-level accuracy, colour consistency |
| **Adversarial (PatchGAN)** | Sharpness and realism |
| **FFT Frequency Loss** | Motivated by SAR speckle physics — forces correct high-frequency texture, which L1 alone averages out |
| **VGG Perceptual Loss** | LPIPS is implemented using pretrained network features; training with VGG loss directly optimises the evaluation metric |

### Ablation Configurations

| Config | Loss | Purpose |
|---|---|---|
| A | L1 only | Baseline |
| B | L1 + Adversarial | + GAN |
| C | L1 + Adversarial + FFT | + Frequency domain |
| **D (main)** | **L1 + Adversarial + FFT + VGG** | **Full model** |

---

## Requirements

- Python 3.10+
- CUDA GPU (≥4 GB VRAM locally; ≤16 GB on Kaggle/Colab)
- See `requirements.txt` for all pinned dependencies

---

## Environment Setup

```bash
# Clone the repository
git clone https://github.com/Trafalgar-2006/sar2eo.git
cd sar2eo

# Create virtual environment
python -m venv venv
source venv/bin/activate       # Linux/Mac
# OR
venv\Scripts\activate          # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Dataset Structure

This project uses the **SEN1-2** dataset (CC-BY 4.0) from TU Munich and/or the **Kaggle Sentinel-1&2 terrain-split** dataset.

### Option 1: SEN1-2
Download via rsync (password: `m1436631`):
```bash
rsync -avz rsync://m1436631@dataserv.ub.tum.de/m1436631/ ./data/SEN1-2/
```

Expected directory layout:
```
data/SEN1-2/
├── ROIs1158_spring/
│   ├── s1_1/          ← SAR grayscale PNGs
│   └── s2_1/          ← EO RGB PNGs
├── ROIs1868_summer/
├── ROIs1970_fall/
└── ROIs2017_winter/
```

### Option 2: Kaggle Terrain-Split
Download from [Kaggle](https://www.kaggle.com/datasets/requiemonk/sentinel12-image-pairs-segregated-by-terrain) and place:
```
data/sentinel12/
├── agri/
│   ├── s1/   ← SAR PNGs
│   └── s2/   ← EO PNGs
├── barrenland/
├── grassland/
└── urban/
```

Update `config.yaml` → `data.dataset_type` to `"sen12"` or `"kaggle"` accordingly.

**Train/Val/Test Split:**
- **SEN1-2**: Split by season — spring+summer+fall=train, winter=val/test (prevents adjacent-patch leakage)
- **Kaggle**: Split by terrain — agri+barrenland+grassland=train, urban=val/test (hardest terrain class held out)

---

## Training

```bash
# Train the full model (Config D — recommended)
python train.py --config config.yaml

# Train a specific ablation config
python train.py --config config.yaml --ablation l1_only     # Config A
python train.py --config config.yaml --ablation l1_adv      # Config B
python train.py --config config.yaml --ablation l1_adv_fft  # Config C
python train.py --config config.yaml --ablation full        # Config D
```

Training logs per-epoch train/val loss to:
- `outputs/loss_curve_{ablation}.png` — visual plot
- `outputs/losses_{ablation}.csv` — raw values (reproducible)

Checkpoints saved to `checkpoints/{ablation}/`:
- `best.pth` — best validation checkpoint
- `epoch_N.pth` — periodic saves (every 10 epochs)
- `final.pth` — last epoch

---

## Inference

Conforms exactly to the GalaxEye I/O contract:

```bash
python infer.py \
  --input_dir  <path/to/sar_patches> \
  --output_dir <path/to/eo_output>   \
  --weights    checkpoints/full/best.pth
```

**Input:** Directory of 256×256 8-bit grayscale PNG, dB-scaled SAR (VV) patches  
**Output:** Directory of 256×256 RGB PNG EO images, same filenames as inputs  
**Constraints:** Single GPU ≤16 GB VRAM, no internet access required

Optional flags:
```bash
--model_config config.yaml   # Path to config (optional)
--device       cuda          # auto | cuda | cpu
--batch_size   8             # Reduce if OOM
```

---

## Evaluation

```bash
# Auto-run inference + compute all metrics on test split
python eval.py \
  --config  config.yaml \
  --weights checkpoints/full/best.pth \
  --split   test

# Or evaluate from existing prediction directories
python eval.py \
  --pred_dir outputs/eval_preds/ \
  --gt_dir   outputs/eval_gt/
```

Computes and saves to `outputs/metrics_{ablation}_{split}.csv`:

| Metric | Type | Direction |
|---|---|---|
| LPIPS | Perceptual | ↓ lower is better |
| FID | Perceptual | ↓ lower is better |
| SSIM | Pixel-level | ↑ higher is better |
| PSNR | Pixel-level | ↑ higher is better |

---

## Model Weights

Pre-trained weights (Config D — full model, trained on Kaggle T4 × 2):

> **[Download weights (Google Drive ~218 MB)](https://drive.google.com/file/d/11Z9o2HNKBPfxBLpSTVbxkHFIMuoFBfBx/view?usp=sharing)**

> ⚠️ This link is also in the Google Form submission. Weights are NOT in the ZIP.

Place checkpoint at: `checkpoints/full/best.pth`

---

## Results

All configs evaluated on the **urban held-out test split (4,000 pairs)**.

### Ablation Study — Test Split

| Config | Loss Function | LPIPS ↓ | FID ↓ | SSIM ↑ | PSNR ↑ |
|---|---|---|---|---|---|
| A: L1 only | L1 | 0.894 | 333.0 | 0.128 | 12.91 |
| B: L1 + Adv | L1 + Adversarial | 0.626 | 317.1 | 0.083 | 12.02 |
| C: L1 + Adv + FFT | L1 + Adv + FFT | 0.627 | 329.1 | 0.078 | 11.65 |
| **D: Full (main)** | **L1 + Adv + FFT + VGG** | **0.615** | **277.9** | 0.073 | 12.22 |

Key findings:
- Adding adversarial loss (B vs A): LPIPS improves 30% (0.894 → 0.626), FID drops from 333 → 317
- Adding VGG perceptual loss (D vs C): FID drops 15% (329 → 278) — largest single gain
- Config D achieves best LPIPS and FID (primary metrics)
- Classic pixel-perceptual tradeoff: L1-only (A) has highest SSIM/PSNR but worst LPIPS/FID

### Training Loss Curve (Config D)

![Loss Curve](outputs/loss_curve_full.png)

---

## Citation / References

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

Zhang et al. (2018). The Unreasonable Effectiveness of Deep Features as a Perceptual Metric.
CVPR 2018. https://arxiv.org/abs/1801.03924

Heusel et al. (2017). GANs Trained by a Two Time-Scale Update Rule Converge to a 
Local Nash Equilibrium. NeurIPS 2017. https://arxiv.org/abs/1706.08500
```

---

## Time & Resource Log

| Phase | Time (approx) |
|---|---|
| Data exploration & literature survey | ~3 hrs |
| Architecture design & implementation | ~5 hrs |
| Debugging & Kaggle setup | ~3 hrs |
| Training Config A — L1 only (100 epochs, T4) | ~1.6 hrs |
| Training Config B — L1 + Adv (100 epochs, T4) | ~4 hrs |
| Training Config C — L1 + Adv + FFT (100 epochs, T4) | ~4 hrs |
| Training Config D — Full (100 epochs, T4) | ~7.5 hrs |
| Evaluation & metric computation | ~2 hrs |
| Report writing | ~4 hrs |
| **Total** | **~34 hrs** |

**Hardware:**
- Training: Kaggle T4 GPU (16 GB VRAM), PyTorch 2.1.0 + CUDA 12.x
- Mixed precision (fp16) via `torch.amp` — ~2× VRAM reduction
- Inference verified on CPU (local)

---

## Submission

ZIP file submitted: `FirstName_LastName_GalaxEye.zip`  
Contains: Technical report PDF, loss curves, qualitative triplet images, time log.  
**Weights NOT in ZIP** — submitted via public link above.
