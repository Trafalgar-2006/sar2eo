"""
losses.py — Loss Functions for SAR-to-EO Translation

Five losses used in this work:

1. L1Loss          — Pixel-wise MAE. Colour accuracy, prevents mode collapse.
                     Targets PSNR.

2. GANLoss         — Adversarial loss (BCEWithLogitsLoss on PatchGAN outputs).
                     Now supports multi-scale discriminator (list of outputs).
                     Encourages sharp, realistic textures. Targets FID.

3. FFTLoss         — L1 loss on 2D Fourier magnitude spectra.
                     MOTIVATION: SAR is dominated by speckle — a multiplicative
                     high-frequency noise process. L1 alone averages it out into
                     blurry outputs. FFT loss explicitly penalises frequency-domain
                     errors, forcing correct texture reproduction at all scales.

4. VGGPerceptualLoss — L1 loss on VGG19 feature maps (relu2_2, relu3_3).
                     MOTIVATION: LPIPS is a pretrained-network feature distance.
                     Training with VGG features directly optimises the LPIPS
                     evaluation metric.

5. MSSSIMLoss      — Multi-Scale Structural Similarity loss.
                     MOTIVATION: SSIM is a secondary evaluation metric. By training
                     with MS-SSIM loss, we directly optimise it. MS-SSIM is better
                     than plain SSIM because it operates at multiple scales, capturing
                     both fine texture and coarse structural agreement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import List, Union


# ---------------------------------------------------------------------------
# 1. L1 Loss (pixel-domain)
# ---------------------------------------------------------------------------

class L1Loss(nn.Module):
    """Standard pixel-wise L1 loss."""

    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss(pred, target)


# ---------------------------------------------------------------------------
# 2. GAN Loss (adversarial) — multi-scale aware
# ---------------------------------------------------------------------------

class GANLoss(nn.Module):
    """
    Adversarial loss using BCEWithLogitsLoss on PatchGAN outputs.

    Supports both:
      - Single discriminator output: a single [B, 1, H, W] tensor
      - Multi-scale discriminator output: a list of [B, 1, H_i, W_i] tensors

    In the multi-scale case, the loss is averaged across all scales.

    No sigmoid in discriminator forward pass — handled numerically here
    for training stability (log-sum-exp trick in BCEWithLogitsLoss).
    """

    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def _loss_single(self, pred: torch.Tensor, is_real: bool) -> torch.Tensor:
        target = torch.ones_like(pred) if is_real else torch.zeros_like(pred)
        return self.loss(pred, target)

    def forward(self, pred: Union[torch.Tensor, List[torch.Tensor]],
                is_real: bool) -> torch.Tensor:
        """
        Args:
            pred:    Single tensor [B, 1, H, W] or list of such tensors
                     (from multi-scale discriminator)
            is_real: True for real images, False for fake
        Returns:
            Scalar adversarial loss
        """
        if isinstance(pred, (list, tuple)):
            losses = [self._loss_single(p, is_real) for p in pred]
            return torch.stack(losses).mean()
        return self._loss_single(pred, is_real)


# ---------------------------------------------------------------------------
# 3. FFT Frequency Loss
# ---------------------------------------------------------------------------

class FFTLoss(nn.Module):
    """
    L1 loss on the 2D Fourier magnitude spectrum.

    Computes FFT of generated and ground-truth images, takes the magnitude,
    and applies L1 loss. Forces the model to match the frequency content of
    the target — not just pixel values.

    Operates on each channel independently, then averages.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   [B, C, H, W] — generated EO image, range [-1, 1]
            target: [B, C, H, W] — ground-truth EO image, range [-1, 1]
        """
        pred_fft   = torch.fft.fft2(pred,   norm="ortho")
        target_fft = torch.fft.fft2(target, norm="ortho")
        pred_mag   = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        return F.l1_loss(pred_mag, target_mag)


# ---------------------------------------------------------------------------
# 4. VGG Perceptual Loss
# ---------------------------------------------------------------------------

class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using pretrained VGG19 feature maps.

    Extracts features at:
      - relu2_2  (low-level: edges, textures)
      - relu3_3  (mid-level: patterns, structures)

    WHY THIS DIRECTLY TARGETS LPIPS:
    LPIPS uses pretrained network features; training with VGG feature loss
    is training a proxy for the evaluation metric.
    """

    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        vgg.eval()
        for p in vgg.parameters():
            p.requires_grad = False

        features = vgg.features
        self.slice1 = nn.Sequential(*list(features.children())[:9])    # relu2_2
        self.slice2 = nn.Sequential(*list(features.children())[9:18])  # relu3_3

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = (x + 1.0) / 2.0          # [-1,1] → [0,1]
        x = (x - self.mean) / self.std
        return x

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_v   = self._preprocess(pred)
        target_v = self._preprocess(target)

        pred_f1   = self.slice1(pred_v)
        target_f1 = self.slice1(target_v)
        pred_f2   = self.slice2(pred_f1)
        target_f2 = self.slice2(target_f1)

        return F.l1_loss(pred_f1, target_f1) + F.l1_loss(pred_f2, target_f2)


# ---------------------------------------------------------------------------
# 5. MS-SSIM Loss (Multi-Scale Structural Similarity)
# ---------------------------------------------------------------------------

class MSSSIMLoss(nn.Module):
    """
    Multi-Scale Structural Similarity loss.

    Reference: Wang et al. (2003). Multi-scale structural similarity for
    image quality assessment. Asilomar Conference on Signals, Systems and
    Computers. https://doi.org/10.1109/ACSSC.2003.1292216

    WHY MS-SSIM vs. SSIM:
      - Plain SSIM is computed at a single fixed scale.
      - MS-SSIM operates at 5 scales (original + 4 downsampled levels) and
        combines luminance at the finest scale with contrast-structure terms
        at all scales. This captures both local texture AND global structure.
      - SSIM is a secondary evaluation metric: training with MS-SSIM directly
        optimises it rather than hoping L1 alone will give good SSIM.

    Args:
        window_size (int):   Gaussian window size. Default 11 (standard).
        data_range  (float): Image value range. Default 2.0 (for [-1,1] input).
        weights     (list):  MS-SSIM scale weights. Standard Simoncelli values.
    """

    _STANDARD_WEIGHTS = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]

    def __init__(self, window_size: int = 11, data_range: float = 2.0,
                 weights: list = None):
        super().__init__()
        self.window_size = window_size
        self.data_range  = data_range
        self.weights     = weights or self._STANDARD_WEIGHTS

        # Pre-compute the Gaussian window (registered as a buffer so it
        # moves to the right device automatically)
        self.register_buffer("_window", self._make_window(window_size))

    @staticmethod
    def _make_window(size: int, sigma: float = 1.5) -> torch.Tensor:
        """1D Gaussian → outer product → 2D Gaussian window [1, 1, size, size]."""
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        return g.outer(g).unsqueeze(0).unsqueeze(0)   # [1, 1, size, size]

    def _ssim_components(self, x: torch.Tensor,
                         y: torch.Tensor) -> tuple:
        """
        Compute luminance and contrast-structure terms at the current scale.

        Returns:
            (luminance, cs): both are [B, C, H', W'] tensors
        """
        C1 = (0.01 * self.data_range) ** 2
        C2 = (0.03 * self.data_range) ** 2

        B, C, H, W = x.shape
        # Expand window to [C, 1, size, size] for depthwise conv
        window = self._window.expand(C, -1, -1, -1)

        pad = self.window_size // 2

        mu_x  = F.conv2d(x, window, padding=pad, groups=C)
        mu_y  = F.conv2d(y, window, padding=pad, groups=C)
        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sigma_x2 = F.conv2d(x * x, window, padding=pad, groups=C) - mu_x2
        sigma_y2 = F.conv2d(y * y, window, padding=pad, groups=C) - mu_y2
        sigma_xy = F.conv2d(x * y, window, padding=pad, groups=C) - mu_xy

        # Numerical stability clamp
        sigma_x2 = sigma_x2.clamp(min=0)
        sigma_y2 = sigma_y2.clamp(min=0)

        luminance = (2 * mu_xy  + C1) / (mu_x2 + mu_y2  + C1)
        cs        = (2 * sigma_xy + C2) / (sigma_x2 + sigma_y2 + C2)

        return luminance, cs

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   [B, C, H, W] generated image, range [-1, 1]
            target: [B, C, H, W] ground-truth image, range [-1, 1]
        Returns:
            MS-SSIM loss (scalar, lower is better).
            Loss = 1 - MS-SSIM so it can be minimised.
        """
        x = pred.float()
        y = target.float()

        n_scales = len(self.weights)
        cs_per_scale = []

        for i in range(n_scales):
            lum, cs = self._ssim_components(x, y)

            if i < n_scales - 1:
                cs_per_scale.append(cs.mean())
                # Downsample by factor of 2 for next scale
                x = F.avg_pool2d(x, kernel_size=2, stride=2)
                y = F.avg_pool2d(y, kernel_size=2, stride=2)
            else:
                # Final scale: luminance × contrast-structure
                cs_per_scale.append((lum * cs).mean())

        # Weighted product across scales (MS-SSIM definition)
        w = torch.tensor(self.weights, device=pred.device, dtype=pred.dtype)
        # Clamp CS values for numerical stability before power
        cs_stack  = torch.stack(cs_per_scale).clamp(min=1e-8)
        ms_ssim   = (cs_stack ** w).prod()

        return 1.0 - ms_ssim   # loss form


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing losses on: {device}")

    pred   = torch.randn(2, 3, 256, 256).to(device)
    target = torch.randn(2, 3, 256, 256).to(device)
    sar    = torch.randn(2, 1, 256, 256).to(device)

    # Fake multi-scale discriminator output (list of 3 tensors)
    d_out_ms = [torch.randn(2, 1, 30, 30).to(device),
                torch.randn(2, 1, 14, 14).to(device),
                torch.randn(2, 1,  6,  6).to(device)]
    # Single-scale output (still works)
    d_out_ss = torch.randn(2, 1, 30, 30).to(device)

    l1   = L1Loss().to(device)
    gan  = GANLoss().to(device)
    fft  = FFTLoss().to(device)
    vgg  = VGGPerceptualLoss().to(device)
    msss = MSSSIMLoss().to(device)

    print(f"L1       loss: {l1(pred, target).item():.4f}")
    print(f"GAN (multi-scale): {gan(d_out_ms, is_real=True).item():.4f}")
    print(f"GAN (single-scale): {gan(d_out_ss, is_real=True).item():.4f}")
    print(f"FFT      loss: {fft(pred, target).item():.4f}")
    print(f"VGG      loss: {vgg(pred, target).item():.4f}")
    print(f"MS-SSIM  loss: {msss(pred, target).item():.4f}  (should be close to 1 for random input)")

    # Verify MS-SSIM is 0 for identical images
    same_loss = msss(pred, pred.clone()).item()
    print(f"MS-SSIM (pred vs pred): {same_loss:.6f}  (should be ~0)")
    assert same_loss < 0.01, f"MS-SSIM should be ~0 for identical images, got {same_loss}"
    print("All losses OK. ✓")
