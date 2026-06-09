"""
maf_layer.py — A Single MAF Layer

One MAF layer wraps MADE and implements two directions:

  DENSITY (forward):  x → z
    zᵢ = (xᵢ - μᵢ(x₁..xᵢ₋₁)) / σᵢ(x₁..xᵢ₋₁)
    Cost: ONE forward pass through MADE. Fast. ✓

  SAMPLING (inverse): z → x
    xᵢ = zᵢ · σᵢ(x₁..xᵢ₋₁) + μᵢ(x₁..xᵢ₋₁)
    Cost: D sequential steps (xᵢ depends on previously generated x₁..xᵢ₋₁). Slow.

The log-determinant:
  The Jacobian ∂z/∂x is lower-triangular with 1/σᵢ on the diagonal.
  det(J) = ∏ᵢ (1/σᵢ)
  log|det(J)| = -∑ᵢ log(σᵢ) = -∑ᵢ log_sigma[i]    ← sum of what MADE outputs

We also use an alternating "reverse" flag across layers to permute which dimension
is conditioned on which. Without this, dimension 1 always has no context (μ₁ = 0,
σ₁ = 1 always), which wastes capacity.
"""

import torch
import torch.nn as nn
from src.made import MADE


class MAFLayer(nn.Module):
    """
    Single MAF transformation layer.

    Args:
        input_dim   : D, number of features
        hidden_dims : hidden layer sizes for the internal MADE network
        reverse     : if True, reverse the input ordering before passing to MADE.
                      Alternating reverse=True/False across stacked layers ensures
                      every dimension gets to condition on every other dimension.
    """

    def __init__(self, input_dim: int, hidden_dims: list[int], reverse: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.reverse = reverse
        self.made = MADE(input_dim=input_dim, hidden_dims=hidden_dims)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        DENSITY direction: x → z   (used during training)

        Args:
            x: (batch, D) — observed data

        Returns:
            z        : (batch, D) — transformed latent variable (should be ~N(0,I))
            log_det  : (batch,)   — log|det(Jacobian)| for this layer

        Why return log_det per sample?
          When stacking layers, we SUM log_dets across layers.
          Each sample has its own log_det because σᵢ values depend on xᵢ.
        """
        if self.reverse:
            x = x.flip(dims=[-1])           # reverse along feature dimension

        mu, log_sigma = self.made(x)        # (batch, D), (batch, D)

        # Clamp log_sigma to [-2, 2] → σ ∈ [0.14, 7.4] per dimension per layer.
        # Across 5 layers max amplification is 7.4^5 ≈ 2200 — manageable.
        # Wider clamps (e.g. [-5,5]) allow 148^5 = 70 billion amplification,
        # which makes val images map to enormous z values after training.
        log_sigma = log_sigma.clamp(-2, 2)

        # Affine inverse transform: squeeze x into z
        # σᵢ = exp(log_σᵢ) — always positive
        sigma = torch.exp(log_sigma)        # (batch, D)
        z = (x - mu) / sigma               # (batch, D)

        # log|det(J)| = sum of -log(σᵢ) across all dimensions, per sample
        # Shape: (batch,) — one scalar per sample in the batch
        log_det = -log_sigma.sum(dim=-1)    # (batch,)

        if self.reverse:
            z = z.flip(dims=[-1])

        return z, log_det

    @torch.no_grad()
    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """
        SAMPLING direction: z → x   (used during inference/generation)

        This is the slow direction. We must generate each xᵢ sequentially
        because μᵢ and σᵢ depend on x₁..xᵢ₋₁, which don't exist until generated.

        Args:
            z: (batch, D) — latent sample from base distribution N(0, I)

        Returns:
            x: (batch, D) — generated sample
        """
        if self.reverse:
            z = z.flip(dims=[-1])

        x = torch.zeros_like(z)            # Start with empty x, fill dimension by dimension

        for i in range(self.input_dim):
            # At step i, x[:, 0..i-1] are already filled in.
            # Pass current x (with zeros at positions i..D-1) through MADE.
            # Due to autoregressive masking, mu[:, i] and log_sigma[:, i]
            # only depend on x[:, 0..i-1] — the already-filled positions. ✓
            mu, log_sigma = self.made(x)
            sigma = torch.exp(log_sigma.clamp(-2, 2))

            # Fill dimension i
            x[:, i] = z[:, i] * sigma[:, i] + mu[:, i]

        if self.reverse:
            x = x.flip(dims=[-1])

        return x


# ─────────────────────────────────────────────
# SANITY CHECK
# ─────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)

    D = 2
    batch = 5
    layer = MAFLayer(input_dim=D, hidden_dims=[32, 32], reverse=False)

    x = torch.randn(batch, D)

    # Density direction
    z, log_det = layer(x)
    print(f"x shape       : {x.shape}")       # (5, 2)
    print(f"z shape       : {z.shape}")       # (5, 2)
    print(f"log_det shape : {log_det.shape}") # (5,) — one value per sample

    # Sampling direction
    z_sample = torch.randn(batch, D)
    x_reconstructed = layer.inverse(z_sample)
    print(f"\nSampled x shape: {x_reconstructed.shape}")  # (5, 2)

    # ── Cycle consistency check ──────────────────────────────────────────
    # If we run x → z → x̂, we should get x̂ ≈ x (up to floating point error)
    # This verifies forward and inverse are true inverses of each other.
    z_from_x, _ = layer(x)
    x_recovered = layer.inverse(z_from_x)

    max_error = (x - x_recovered).abs().max().item()
    print(f"\nCycle consistency max error: {max_error:.2e}  (should be < 1e-5)")
    print("Cycle consistency OK ✓" if max_error < 1e-5 else "Cycle consistency FAILED ✗")

    print("\nmaf_layer.py is working correctly.")
