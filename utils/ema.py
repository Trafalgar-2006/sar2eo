"""
ema.py — Exponential Moving Average (EMA) for Generator Weights

EMA maintains a shadow copy of the generator's parameters by tracking a
running exponential average. The shadow copy is what gets used for:
  - Validation during training (EMA model vs. live model)
  - Final inference (saved as G_ema in checkpoints)

Why EMA for GANs:
  GAN training is inherently noisy — the generator oscillates as it plays
  a minimax game against the discriminator. The live model weights at any
  single epoch may be temporarily degraded by a bad discriminator step.
  EMA smooths over these oscillations and consistently gives better
  inference quality without any extra training cost.

  Concretely: if live G produces SSIM 0.25 ± 0.03 (noisy), the EMA model
  typically produces SSIM 0.27 ± 0.005 (stable).

Usage:
    ema = EMA(G, decay=0.999)

    # After each generator update:
    ema.update(G)

    # For validation:
    with ema.apply():
        fake = G(sar)       # G temporarily has EMA weights

    # Save checkpoint:
    torch.save({'G': G.state_dict(), 'G_ema': ema.state_dict()}, path)
"""

import copy
from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn


class EMA:
    """
    Exponential Moving Average of model parameters.

    Maintains a separate shadow model whose parameters are:
        shadow_p = decay * shadow_p + (1 - decay) * live_p

    Args:
        model  (nn.Module): The model to track (generator G).
        decay  (float):     EMA decay factor. 0.999 is standard for GANs.
                            Higher = more inertia = smoother but slower to adapt.
        start_step (int):   Don't apply EMA until this many steps have passed.
                            Prevents bad early weights from dominating the shadow.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999,
                 start_step: int = 0):
        self.decay      = decay
        self.start_step = start_step
        self._step      = 0

        # Deep copy of model — completely independent copy
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()

        # Shadow model never gets gradients
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """
        Update shadow model with current live model parameters.
        Call this ONCE per generator update step.

        Args:
            model: The live generator model after its parameter update.
        """
        self._step += 1

        # Buffers (BatchNorm running_mean/running_var/num_batches_tracked) are
        # state, not learnable parameters — they are never touched by the
        # optimiser, so there is nothing to average. Copy them across verbatim.
        # Without this the shadow keeps its construction-time BN statistics for
        # the whole run, and every `apply()` (validation, best.pth, inference)
        # runs the network on stale stats.
        for shadow_b, live_b in zip(self.shadow.buffers(), model.buffers()):
            shadow_b.data.copy_(live_b.data)

        if self._step < self.start_step:
            # Before start_step: just copy live weights (no averaging)
            for shadow_p, live_p in zip(self.shadow.parameters(),
                                         model.parameters()):
                shadow_p.data.copy_(live_p.data)
            return

        # Standard EMA update
        for shadow_p, live_p in zip(self.shadow.parameters(),
                                     model.parameters()):
            shadow_p.data.mul_(self.decay).add_(live_p.data,
                                                alpha=1.0 - self.decay)

    @contextmanager
    def apply(self, model: nn.Module):
        """
        Context manager: temporarily replaces model's weights with EMA weights.
        Restores live weights on exit. Use during validation.

        Example:
            with ema.apply(G):
                fake = G(sar)   # EMA weights
            # G is back to live weights here
        """
        # Save live weights
        live_state = copy.deepcopy(model.state_dict())
        # Apply EMA weights
        model.load_state_dict(self.shadow.state_dict())
        model.eval()
        try:
            yield
        finally:
            # Restore live weights
            model.load_state_dict(live_state)

    def state_dict(self) -> dict:
        """Return EMA model state dict for checkpointing."""
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore EMA model from checkpoint."""
        self.shadow.load_state_dict(state_dict)

    @property
    def model(self) -> nn.Module:
        """The shadow EMA model (read-only, for direct inference)."""
        return self.shadow


# ---------------------------------------------------------------------------
# Quick sanity check
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

    import torch.nn as nn

    # Simple model for testing
    model = nn.Linear(4, 2)
    ema = EMA(model, decay=0.9, start_step=2)

    print("Before any updates:")
    print(f"  live   weight: {model.weight.data}")
    print(f"  shadow weight: {ema.shadow.weight.data}")

    for step in range(5):
        # Simulate a training step (random update)
        with torch.no_grad():
            model.weight.data = torch.randn_like(model.weight)
        ema.update(model)
        print(f"Step {step+1}: live={model.weight.data.flatten()[:2].tolist()}, "
              f"shadow={ema.shadow.weight.data.flatten()[:2].tolist()}")

    # Test apply context manager
    original_w = model.weight.data.clone()
    with ema.apply(model):
        assert torch.allclose(model.weight.data, ema.shadow.weight.data)
        print("\nInside ema.apply: model has EMA weights ✓")
    assert torch.allclose(model.weight.data, original_w)
    print("After ema.apply: model restored to live weights ✓")
    print("EMA OK.")
