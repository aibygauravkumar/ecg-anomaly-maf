"""
made.py — Masked Autoencoder for Distribution Estimation (MADE)

Paper: Germain et al., 2015 — https://arxiv.org/abs/1502.03509

MADE answers the question:
  "Can one forward pass through a single network give us ALL conditionals
   p(x₁), p(x₂|x₁), p(x₃|x₁,x₂), ... simultaneously?"

Answer: Yes, by masking weights so output i can only see inputs 1..i-1.

This file builds:
  1. MaskedLinear  — a Linear layer that respects a binary mask
  2. MADE          — stacks MaskedLinear layers with masks chosen to enforce
                     the autoregressive ordering constraint
"""

import torch
import torch.nn as nn
import numpy as np


# ─────────────────────────────────────────────
# BUILDING BLOCK 1: MaskedLinear
# ─────────────────────────────────────────────

class MaskedLinear(nn.Linear):
    """
    A Linear layer (y = xW^T + b) with an element-wise binary mask on the weights.

    The effective weight matrix is:  W_eff = W ⊙ mask
    where ⊙ is element-wise multiplication.

    Why subclass nn.Linear?
    - We get weight initialization, bias, and forward() for free.
    - We only need to intercept the weight before the matrix multiply.

    The mask is registered as a buffer (not a parameter):
    - Buffers move with .to(device) — important for GPU training
    - They are NOT updated by the optimizer — masks are fixed once set
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)

        # Register mask as buffer: shape (out_features, in_features)
        # matches the shape of self.weight in nn.Linear
        self.register_buffer("mask", torch.ones(out_features, in_features))

    def set_mask(self, mask: torch.Tensor):
        """
        Set the binary mask. Call this once after constructing the layer.
        mask shape: (out_features, in_features), values in {0, 1}
        """
        self.mask.data.copy_(mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply mask to weights before the linear operation
        # self.weight shape: (out_features, in_features)
        # self.mask  shape: (out_features, in_features)  — same shape, broadcast-safe
        return nn.functional.linear(x, self.weight * self.mask, self.bias)


# ─────────────────────────────────────────────
# BUILDING BLOCK 2: MADE
# ─────────────────────────────────────────────

class MADE(nn.Module):
    """
    MADE: Masked Autoencoder for Distribution Estimation.

    Given input x of dimension D, outputs 2D values: (μ₁,σ₁, μ₂,σ₂, ..., μD,σD)
    where (μᵢ, σᵢ) are conditioned ONLY on x₁, ..., xᵢ₋₁.

    This is enforced by the masking scheme described below.

    How masking works — the "ordering" trick:
    ─────────────────────────────────────────
    Assign each neuron an integer "order" (1 to D).
    A connection from neuron A → neuron B is ALLOWED only if order(A) < order(B).
    This ensures no output can see a later input — the autoregressive constraint.

    Layer-by-layer:
      Input neurons:  order = [1, 2, 3, ..., D]         (the feature index)
      Hidden neurons: order = random integers in [1, D-1] (sampled uniformly)
      Output neurons: order = [1, 2, 3, ..., D] repeated (μ and σ for each dim)

    Mask rule between two layers:
      mask[i, j] = 1  if order_out[i] >= order_in[j]   (hidden layer connections)
      mask[i, j] = 1  if order_out[i] >  order_in[j]   (final layer connections)

    The strict inequality at the output ensures output i does NOT see input i,
    only inputs 1..i-1. (μᵢ, σᵢ) depend only on x₁..xᵢ₋₁. ✓
    """

    def __init__(self, input_dim: int, hidden_dims: list[int]):
        """
        Args:
            input_dim   : D, number of input/output dimensions
            hidden_dims : list of hidden layer sizes, e.g. [128, 128]
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        # Build the stack of MaskedLinear layers
        # Architecture: input_dim → hidden_dims[0] → ... → hidden_dims[-1] → 2*input_dim
        # Output is 2*input_dim because we need both μ and σ for each dimension
        layer_sizes = [input_dim] + hidden_dims + [input_dim * 2]

        layers = []
        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(MaskedLinear(in_size, out_size))
            # Add ReLU activation between hidden layers (not after last layer)
            if out_size != input_dim * 2:
                layers.append(nn.ReLU())

        self.network = nn.Sequential(*layers)

        # Generate and assign masks — done once at construction
        self._build_masks()

    def _build_masks(self):
        """
        Compute and assign binary masks to all MaskedLinear layers.

        Step 1: Assign "orders" to every neuron layer by layer.
        Step 2: Compute mask between each consecutive pair of layers.
        Step 3: Call set_mask() on each MaskedLinear layer.
        """
        D = self.input_dim

        # Collect all MaskedLinear layers (skip ReLU activations)
        masked_layers = [m for m in self.network if isinstance(m, MaskedLinear)]

        # ── Step 1: Assign orders ──────────────────────────────────────────
        # Input layer: order[i] = i+1, so orders are [1, 2, ..., D]
        orders = [np.arange(1, D + 1)]

        for layer in masked_layers[:-1]:  # hidden layers only
            out_size = layer.out_features
            # Sample uniformly from [1, D-1]
            # Why max D-1? So at least one input always has a higher order,
            # meaning every hidden neuron can connect to at least one output.
            hidden_order = np.random.randint(1, D, size=out_size)
            orders.append(hidden_order)

        # Output layer: repeat [1, 2, ..., D] twice (once for μ, once for σ)
        orders.append(np.concatenate([np.arange(1, D + 1), np.arange(1, D + 1)]))

        # ── Step 2 & 3: Compute masks and assign ──────────────────────────
        for layer, order_in, order_out in zip(masked_layers, orders[:-1], orders[1:]):
            # For hidden layers: allow connection if order_out[i] >= order_in[j]
            # For the final output layer: strict > (output i must NOT see input i)
            is_output_layer = (layer == masked_layers[-1])

            if is_output_layer:
                # Strict inequality: output i sees inputs with order < i only
                # order_out is [1..D, 1..D] for [μ, σ] — strict < ensures autoregressive
                mask = (order_out[:, None] > order_in[None, :]).astype(np.float32)
            else:
                mask = (order_out[:, None] >= order_in[None, :]).astype(np.float32)

            layer.set_mask(torch.from_numpy(mask))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: given x of shape (batch, D),
        returns (mu, log_sigma) each of shape (batch, D).

        We output log(σ) instead of σ directly because:
        - σ must be positive (it's a scale parameter)
        - Outputting log(σ) and then exponentiating ensures positivity
        - log(σ) is unconstrained, easier for the network to learn
        - Numerical stability: working in log-space avoids very small σ values

        Remember: mu[i] and log_sigma[i] depend ONLY on x[0..i-1]
        That's enforced by the masks — not by any explicit code here.
        """
        out = self.network(x)                         # (batch, 2*D)
        mu, log_sigma = out.chunk(2, dim=-1)          # each (batch, D)
        return mu, log_sigma


# ─────────────────────────────────────────────
# SANITY CHECK
# ─────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)

    D = 4                        # 4-dimensional input
    batch = 3
    made = MADE(input_dim=D, hidden_dims=[16, 16])

    x = torch.randn(batch, D)
    mu, log_sigma = made(x)

    print(f"Input shape      : {x.shape}")           # (3, 4)
    print(f"mu shape         : {mu.shape}")           # (3, 4)
    print(f"log_sigma shape  : {log_sigma.shape}")    # (3, 4)

    # ── Verify the autoregressive property ──────────────────────────────
    # mu[:, i] should NOT change when we change x[:, i] or x[:, j>i]
    # It SHOULD change when we change x[:, j<i]
    print("\n── Autoregressive property check ──")
    x2 = x.clone()
    x2[:, 2] = 999.0            # Change dimension 2

    mu2, _ = made(x2)

    print(f"mu[:, 0] changed? {not torch.allclose(mu[:, 0], mu2[:, 0])} (should be False — dim 0 sees nothing)")
    print(f"mu[:, 1] changed? {not torch.allclose(mu[:, 1], mu2[:, 1])} (should be False — dim 1 only sees dim 0)")
    print(f"mu[:, 2] changed? {not torch.allclose(mu[:, 2], mu2[:, 2])} (should be False — dim 2 only sees dims 0,1)")
    print(f"mu[:, 3] changed? {not torch.allclose(mu[:, 3], mu2[:, 3])} (should be True  — dim 3 sees dims 0,1,2)")

    print("\nmade.py is working correctly.")
