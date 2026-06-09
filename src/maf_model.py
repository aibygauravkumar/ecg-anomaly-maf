"""
maf_model.py — Full Stacked MAF Model

Stacks N MAF layers. Between layers, we alternate the ordering direction
(reverse=True/False). This ensures that across the full model, every
dimension gets a chance to condition on every other dimension.

Why alternating reversal matters:
  Layer 1 (normal):  x₁ conditions on nothing, x₂ conditions on x₁
  Layer 2 (reverse): x₂ conditions on nothing, x₁ conditions on x₂
  Layer 3 (normal):  now x₁ has info from x₂ baked in from layer 2

Without alternating, dimension 1 always starts with zero context — it
would be forced to rely entirely on the base distribution for its first
dimension, reducing model expressiveness.

The full log-likelihood formula for a data point x through K layers:

  log p(x) = log p_z(z_K)  +  Σₖ log|det(Jₖ)|

where z_K is the final latent after all K transformations, and each
layer contributes its own log-determinant term.
"""

import torch
import torch.nn as nn
from src.maf_layer import MAFLayer


class MAF(nn.Module):
    """
    Stacked Masked Autoregressive Flow.

    Args:
        input_dim   : D, dimensionality of the data
        n_layers    : number of MAF layers to stack
        hidden_dims : hidden layer sizes for each MADE sub-network
    """

    def __init__(self, input_dim: int, n_layers: int = 5, hidden_dims: list[int] = [128, 128]):
        super().__init__()

        self.input_dim = input_dim

        # Stack layers with alternating reversal
        self.layers = nn.ModuleList([
            MAFLayer(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                reverse=(i % 2 == 1)        # odd layers are reversed
            )
            for i in range(n_layers)
        ])

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log p(x) for a batch of data points.

        This is what we maximize during training (negative = loss).

        Flow:  x  →  z₁  →  z₂  →  ...  →  z_K  ~  N(0, I)
        Each →  is one MAF layer in the density direction.

        Args:
            x: (batch, D)

        Returns:
            log_prob: (batch,) — log-likelihood of each sample
        """
        total_log_det = torch.zeros(x.shape[0], device=x.device)

        z = x
        for layer in self.layers:
            z, log_det = layer(z)
            total_log_det += log_det        # accumulate log|det| across all layers

        # z is now (approximately) Gaussian if the model is trained well
        # Evaluate log p_z(z) under N(0, I):
        #   log N(z; 0, I) = -D/2 · log(2π) - 1/2 · ||z||²
        log_base = self._log_gaussian(z)    # (batch,)

        return log_base + total_log_det

    def _log_gaussian(self, z: torch.Tensor) -> torch.Tensor:
        """
        Log-likelihood of z under the standard multivariate Gaussian N(0, I).

          log N(z; 0, I) = -D/2 · log(2π) - 1/2 · Σᵢ zᵢ²

        Args:
            z: (batch, D)
        Returns:
            (batch,) — one scalar per sample
        """
        D = z.shape[-1]
        log_2pi = torch.log(torch.tensor(2 * torch.pi))
        return -0.5 * (D * log_2pi + (z ** 2).sum(dim=-1))

    @torch.no_grad()
    def sample(self, n_samples: int, device: str = "cpu") -> torch.Tensor:
        """
        Generate new samples by running the model backwards: N(0,I) → data.

        Flow:  z_K  →  z_{K-1}  →  ...  →  z₁  →  x
        Each → is one MAF layer in the sampling (inverse) direction.

        Args:
            n_samples : how many samples to generate
            device    : "cpu" or "cuda"

        Returns:
            x: (n_samples, D) — generated data samples
        """
        z = torch.randn(n_samples, self.input_dim, device=device)

        # Run layers in REVERSE order for inverse direction
        for layer in reversed(self.layers):
            z = layer.inverse(z)

        return z


# ─────────────────────────────────────────────
# SANITY CHECK
# ─────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)

    D = 2           # 2D data (matching our two-moons dataset)
    model = MAF(input_dim=D, n_layers=5, hidden_dims=[64, 64])

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Test log_prob
    x = torch.randn(10, D)
    log_p = model.log_prob(x)
    print(f"\nlog_prob shape  : {log_p.shape}")       # (10,)
    print(f"log_prob values : {log_p[:3].detach()}")  # should be finite numbers

    # Test sampling
    samples = model.sample(n_samples=100)
    print(f"\nSamples shape   : {samples.shape}")     # (100, 2)
    print(f"Samples mean    : {samples.mean(dim=0)}")
    print(f"Samples std     : {samples.std(dim=0)}")

    # Before training, samples should look ~Gaussian (model hasn't learned anything yet)
    print("\nmaf_model.py is working correctly.")
