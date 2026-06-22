"""
losses.py — Loss Functions for SAR-to-EO Translation

Four losses used in this work:

1. L1Loss         — Pixel-wise MAE. Ensures colour/brightness accuracy.
                    Targets PSNR and SSIM.

2. GANLoss        — Adversarial loss (BCEWithLogitsLoss on PatchGAN output).
                    Encourages sharp, realistic textures. Targets FID.

3. FFTLoss        — L1 loss on Fourier magnitude spectra.
                    MOTIVATION: SAR images are dominated by speckle — a
                    multiplicative high-frequency noise process. Training
                    with only pixel-domain L1 loss causes the generator to
                    produce over-smoothed (blurry) outputs that average
                    out high-frequency uncertainty. The FFT loss explicitly
                    penalises errors in the frequency domain, forcing the
                    model to reproduce high-frequency texture (edges, fine
                    structure) correctly — exactly what LPIPS and FID reward.

4. VGGPerceptualLoss — L1 loss on VGG19 feature maps (relu2_2, relu3_3).
                    MOTIVATION: LPIPS (our primary metric) is literally a
                    learned perceptual similarity measure implemented using
                    pretrained network features. Training with VGG feature
                    loss is thus directly optimising the evaluation metric.
                    Targets semantic texture similarity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


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
# 2. GAN Loss (adversarial)
# ---------------------------------------------------------------------------

class GANLoss(nn.Module):
    """
    Adversarial loss using BCEWithLogitsLoss on PatchGAN outputs.

    No sigmoid in discriminator forward pass — handled numerically here
    for training stability (log-sum-exp trick).

    Usage:
        gan_loss = GANLoss()
        # Discriminator real loss:
        loss_D_real = gan_loss(D(sar, real_eo), is_real=True)
        # Discriminator fake loss:
        loss_D_fake = gan_loss(D(sar, fake_eo.detach()), is_real=False)
        # Generator adversarial loss:
        loss_G_adv  = gan_loss(D(sar, fake_eo), is_real=True)  # fool D
    """

    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, pred: torch.Tensor, is_real: bool) -> torch.Tensor:
        target = torch.ones_like(pred) if is_real else torch.zeros_like(pred)
        return self.loss(pred, target)


# ---------------------------------------------------------------------------
# 3. FFT Frequency Loss
# ---------------------------------------------------------------------------

class FFTLoss(nn.Module):
    """
    L1 loss on the 2D Fourier magnitude spectrum.

    Computes FFT of generated and ground-truth images, takes the magnitude
    (amplitude), and applies L1 loss. This forces the model to match the
    frequency content of the target — not just the pixel values.

    Operates on each channel independently, then averages.

    Reference: Motivated by frequency-domain image analysis in SAR processing;
    related to spectral losses in super-resolution literature.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   [B, C, H, W] — generated EO image, range [-1, 1]
            target: [B, C, H, W] — ground-truth EO image, range [-1, 1]
        Returns:
            Scalar FFT magnitude L1 loss.
        """
        # Compute 2D FFT (complex)
        pred_fft   = torch.fft.fft2(pred,   norm="ortho")
        target_fft = torch.fft.fft2(target, norm="ortho")

        # Magnitude (amplitude) spectrum
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

    Uses L1 distance between feature maps of pred and target.

    The VGG19 weights are frozen (eval mode, no gradient).

    WHY THIS DIRECTLY TARGETS LPIPS:
    LPIPS (Learned Perceptual Image Patch Similarity) is implemented using
    pretrained network features — it computes a weighted sum of feature-map
    differences between images. By training with VGG feature loss, we are
    directly optimising a proxy for the LPIPS evaluation metric.
    """

    def __init__(self):
        super().__init__()

        # Load pretrained VGG19, freeze weights
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        vgg.eval()
        for p in vgg.parameters():
            p.requires_grad = False

        features = vgg.features

        # relu2_2 → features[0:9] (indices 0..8)
        # relu3_3 → features[0:18] (indices 0..17)
        # Using sequential slices to extract at multiple depths
        self.slice1 = nn.Sequential(*list(features.children())[:9])   # up to relu2_2
        self.slice2 = nn.Sequential(*list(features.children())[9:18]) # relu2_2 → relu3_3

        # ImageNet normalisation (VGG was trained on ImageNet)
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Rescale from [-1,1] to [0,1], then apply ImageNet normalisation."""
        x = (x + 1.0) / 2.0                   # [-1,1] → [0,1]
        x = (x - self.mean) / self.std         # ImageNet normalise
        return x

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   [B, 3, H, W] — generated EO, range [-1, 1]
            target: [B, 3, H, W] — ground-truth EO, range [-1, 1]
        Returns:
            Scalar perceptual loss.
        """
        pred_v   = self._preprocess(pred)
        target_v = self._preprocess(target)

        # Extract features at slice1 and slice2
        pred_f1   = self.slice1(pred_v)
        target_f1 = self.slice1(target_v)

        pred_f2   = self.slice2(pred_f1)
        target_f2 = self.slice2(target_f1)

        loss = F.l1_loss(pred_f1, target_f1) + F.l1_loss(pred_f2, target_f2)
        return loss


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing losses on: {device}")

    pred   = torch.randn(2, 3, 256, 256).to(device)
    target = torch.randn(2, 3, 256, 256).to(device)
    sar    = torch.randn(2, 1, 256, 256).to(device)
    # Fake discriminator output
    d_out  = torch.randn(2, 1, 30, 30).to(device)

    l1  = L1Loss().to(device)
    gan = GANLoss().to(device)
    fft = FFTLoss().to(device)
    vgg = VGGPerceptualLoss().to(device)

    print(f"L1  loss: {l1(pred, target).item():.4f}")
    print(f"GAN loss: {gan(d_out, is_real=True).item():.4f}")
    print(f"FFT loss: {fft(pred, target).item():.4f}")
    print(f"VGG loss: {vgg(pred, target).item():.4f}")
    print("All losses OK.")
