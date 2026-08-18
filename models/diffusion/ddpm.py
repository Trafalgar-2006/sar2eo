"""
models/diffusion/ddpm.py — DDPM + DDIM Diffusion Scheduler

Implements:
  - Cosine noise schedule (better than linear for images)
  - Forward process: q(x_t | x_0)
  - Reverse process: p_θ(x_{t-1} | x_t, SAR)  — model step
  - DDIM fast sampler (50 steps instead of 1000)
  - Metrics: computes SNR for training loss weighting

Usage:
    ddpm  = DDPM(timesteps=1000)
    ddim  = DDIMSampler(ddpm, ddim_steps=50)

    # Training:
    noise = torch.randn_like(x0)
    t     = torch.randint(0, ddpm.T, (B,))
    x_t   = ddpm.q_sample(x0, t, noise)
    pred  = model(x_t, sar, t)
    loss  = ddpm.loss(pred, noise, t)   # SNR-weighted L2

    # Inference:
    x_gen = ddim.sample(model, sar, shape=(B, 3, 256, 256))
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Noise schedule
# ---------------------------------------------------------------------------

def cosine_noise_schedule(timesteps: int, s: float = 0.008) -> dict:
    """
    Cosine noise schedule from Nichol & Dhariwal (2021).
    Better than linear — avoids too-noisy images at the end of the schedule.

    Returns dict of pre-computed schedule tensors.
    """
    t  = torch.arange(timesteps + 1, dtype=torch.float64)
    f  = torch.cos(((t / timesteps) + s) / (1 + s) * np.pi / 2) ** 2
    alphas_cumprod = f / f[0]
    betas          = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas          = betas.clamp(max=0.999)
    alphas         = 1 - betas
    alphas_cumprod = alphas.cumprod(dim=0)
    alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

    return {
        "betas":                 betas.float(),
        "alphas":                alphas.float(),
        "alphas_cumprod":        alphas_cumprod.float(),
        "alphas_cumprod_prev":   alphas_cumprod_prev.float(),
        "sqrt_alphas_cumprod":   alphas_cumprod.sqrt().float(),
        "sqrt_one_minus_ac":     (1 - alphas_cumprod).sqrt().float(),
        "log_one_minus_ac":      (1 - alphas_cumprod).log().float(),
        "sqrt_recip_ac":         (1 / alphas_cumprod).sqrt().float(),
        "sqrt_recipm1_ac":       (1 / alphas_cumprod - 1).sqrt().float(),
        "posterior_var":         (
            betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        ).float(),
    }


def _extract(schedule_tensor: torch.Tensor, t: torch.Tensor,
             shape: tuple) -> torch.Tensor:
    """Gather schedule values at timesteps t and reshape to match x."""
    out = schedule_tensor.to(t.device)[t]
    return out.view(t.shape[0], *((1,) * (len(shape) - 1)))


# ---------------------------------------------------------------------------
# DDPM
# ---------------------------------------------------------------------------

class DDPM(nn.Module):
    """
    Denoising Diffusion Probabilistic Model scheduler.

    Handles forward process (noising) and reverse process (denoising).
    The actual neural network is external (ConditionalUNet).

    Prediction mode:
        "eps"  — model predicts noise ε       (standard, recommended)
        "x0"   — model predicts clean x_0     (sometimes more stable)
    """

    def __init__(self, timesteps: int = 1000, pred_mode: str = "eps"):
        super().__init__()
        assert pred_mode in ("eps", "x0")
        self.T         = timesteps
        self.pred_mode = pred_mode

        sched = cosine_noise_schedule(timesteps)
        for k, v in sched.items():
            self.register_buffer(k, v)

    # ── Forward process (training) ──────────────────────────────────────────
    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor,
                 noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Sample x_t from q(x_t | x_0) = N(√ᾱ_t x_0, (1-ᾱ_t)I)

        Args:
            x_0   : [B, C, H, W] clean image in [-1, 1]
            t     : [B] integer timesteps
            noise : optional noise (uses randn if None)

        Returns:
            x_t : [B, C, H, W] noisy image at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        s = x_0.shape
        mean  = _extract(self.sqrt_alphas_cumprod,    t, s) * x_0
        sigma = _extract(self.sqrt_one_minus_ac,      t, s) * noise
        return mean + sigma, noise

    # ── SNR-weighted loss (improves training stability at high t) ───────────
    def loss(self, pred: torch.Tensor, target: torch.Tensor,
             t: torch.Tensor) -> torch.Tensor:
        """
        Min-SNR weighted L2 loss (Hang et al. 2023).
        Upweights high-t (noisy) timesteps which are underweighted by default.

        Args:
            pred   : model output
            target : noise ε (or x_0 in x0 mode)
            t      : [B] timesteps

        Returns:
            scalar loss
        """
        # Compute per-sample MSE
        mse = ((pred - target) ** 2).flatten(1).mean(1)   # [B]

        # SNR = ᾱ_t / (1 - ᾱ_t)
        ac    = _extract(self.alphas_cumprod, t, t.shape)
        snr   = ac / (1 - ac + 1e-8)                      # [B]

        # Min-SNR-γ weighting (γ=5 recommended)
        gamma = 5.0
        w     = torch.minimum(snr, torch.full_like(snr, gamma)) / gamma
        return (w * mse).mean()

    # ── Reverse process (inference) ─────────────────────────────────────────
    @torch.no_grad()
    def p_sample(self, model, x_t: torch.Tensor, sar: torch.Tensor,
                 t: int) -> torch.Tensor:
        """
        One step of the reverse process: sample x_{t-1} from p_θ(x_{t-1}|x_t).
        Uses ancestral sampling (DDPM).
        """
        t_tensor = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)

        # Model prediction
        pred = model(x_t, sar, t_tensor)

        if self.pred_mode == "eps":
            # Reconstruct x_0 from predicted noise
            x0_pred = (
                _extract(self.sqrt_recip_ac, t_tensor, x_t.shape) * x_t
                - _extract(self.sqrt_recipm1_ac, t_tensor, x_t.shape) * pred
            ).clamp(-1, 1)
        else:
            x0_pred = pred.clamp(-1, 1)

        # Posterior mean
        betas      = _extract(self.betas, t_tensor, x_t.shape)
        ac         = _extract(self.alphas_cumprod, t_tensor, x_t.shape)
        ac_prev    = _extract(self.alphas_cumprod_prev, t_tensor, x_t.shape)
        post_mean  = (
            (ac_prev.sqrt() * betas) / (1 - ac) * x0_pred
            + ((1 - ac_prev) * (1 - betas).sqrt()) / (1 - ac) * x_t
        )
        post_var   = _extract(self.posterior_var, t_tensor, x_t.shape)

        noise = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)
        return post_mean + post_var.sqrt() * noise

    @torch.no_grad()
    def p_sample_loop(self, model, sar: torch.Tensor,
                      shape: tuple) -> torch.Tensor:
        """Full reverse loop: x_T → x_0 using DDPM (1000 steps). Slow."""
        device = sar.device
        x = torch.randn(shape, device=device)
        for t in reversed(range(self.T)):
            x = self.p_sample(model, x, sar, t)
        return x


# ---------------------------------------------------------------------------
# DDIM — fast sampler (50 steps instead of 1000)
# ---------------------------------------------------------------------------

class DDIMSampler:
    """
    Denoising Diffusion Implicit Models (Song et al., 2020).
    Same quality as DDPM but in 10–50 steps instead of 1000.

    This is the sampler used for inference.
    """

    def __init__(self, ddpm: DDPM, ddim_steps: int = 50, eta: float = 0.0):
        """
        Args:
            ddpm       : trained DDPM (provides schedule)
            ddim_steps : number of inference steps (20–50 is good)
            eta        : 0 = deterministic DDIM, 1 = stochastic DDPM
        """
        self.ddpm  = ddpm
        self.steps = ddim_steps
        self.eta   = eta

        # Sub-sample timesteps evenly
        T    = ddpm.T
        step = T // ddim_steps
        self.timesteps = list(reversed(range(0, T, step)))[:ddim_steps]

    @torch.no_grad()
    def sample(self, model, sar: torch.Tensor,
               shape: tuple) -> torch.Tensor:
        """
        Generate EO image from SAR condition using DDIM.

        Args:
            model  : ConditionalUNet (or any denoiser with same signature)
            sar    : [B, 1, H, W] SAR condition
            shape  : output shape, e.g. (B, 3, 256, 256)

        Returns:
            [B, 3, H, W] generated EO in [-1, 1]
        """
        device = sar.device
        x = torch.randn(shape, device=device)

        for i, t_cur in enumerate(self.timesteps):
            t_prev = self.timesteps[i + 1] if i + 1 < len(self.timesteps) else 0

            t_tensor = torch.full((shape[0],), t_cur,
                                  device=device, dtype=torch.long)

            # Model prediction
            pred = model(x, sar, t_tensor)

            ac      = self.ddpm.alphas_cumprod[t_cur]
            ac_prev = self.ddpm.alphas_cumprod[t_prev]

            if self.ddpm.pred_mode == "eps":
                # Predict x_0 from ε
                x0_pred = (x - (1 - ac).sqrt() * pred) / ac.sqrt()
            else:
                x0_pred = pred
            x0_pred = x0_pred.clamp(-1, 1)

            # DDIM update
            sigma  = (self.eta
                      * ((1 - ac_prev) / (1 - ac)).sqrt()
                      * (1 - ac / ac_prev).sqrt())
            dir_xt = (1 - ac_prev - sigma ** 2).sqrt() * pred
            noise  = sigma * torch.randn_like(x) if self.eta > 0 else 0

            x = ac_prev.sqrt() * x0_pred + dir_xt + noise

        return x.clamp(-1, 1)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Windows consoles default to cp1252 and raise on the unicode used in the
    # progress output below. Force UTF-8 so local runs match Kaggle/Linux.
    import sys as _sys
    try:
        _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    ddpm = DDPM(timesteps=1000)
    print(f"Schedule tensors registered: {[k for k in ddpm._buffers]}")

    B = 2
    x0 = torch.randn(B, 3, 256, 256)
    t  = torch.randint(0, 1000, (B,))
    xt, noise = ddpm.q_sample(x0, t)
    print(f"q_sample: x0={x0.shape} → xt={xt.shape}")
    print(f"xt range: [{xt.min():.2f}, {xt.max():.2f}]")

    dummy_pred = torch.randn_like(xt)
    loss = ddpm.loss(dummy_pred, noise, t)
    print(f"SNR-weighted loss: {loss.item():.4f}")

    sampler = DDIMSampler(ddpm, ddim_steps=50)
    print(f"DDIM timestep count: {len(sampler.timesteps)}")
    print(f"DDIM timesteps (first 5): {sampler.timesteps[:5]}")
