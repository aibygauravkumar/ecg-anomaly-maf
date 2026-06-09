"""
src/encoder.py — 1D CNN Encoder (the "Eyes" of the hybrid model)

Converts a raw 187-point ECG signal into a compact, noise-robust
feature vector of size `latent_dim` (default 32).

Why CNN before MAF?
  Raw ECG signals have three problems for normalizing flows:
    1. High-frequency noise       (muscle artifacts, electrical interference)
    2. Baseline wander            (slow drift caused by breathing)
    3. Time misalignment          (R-peak not always at exactly the same position)

  A CNN handles all three naturally:
    - Conv filters smooth out high-frequency noise
    - Multiple pooling layers remove slow drift
    - Sliding windows are position-invariant (same filter works anywhere)

  After encoding, the MAF receives a stable 32-number vector
  instead of a noisy 187-number signal. This is why:
    - MAF scores stay in a sensible range
    - Training is stable
    - The model generalises to new recordings
"""

import torch
import torch.nn as nn


class ECGEncoder(nn.Module):
    """
    1D CNN that compresses a 187-point heartbeat into a latent_dim vector.

    Architecture:
      Input:  (batch, 187)  — raw normalised ECG signal

      Conv block 1: learns local wave patterns (P, Q, R, S, T waves)
        Conv1D(1→16, kernel=7) → ReLU → MaxPool(2) → shape (batch, 16, 93)

      Conv block 2: learns beat-level structure
        Conv1D(16→32, kernel=5) → ReLU → MaxPool(2) → shape (batch, 32, 46)

      Conv block 3: learns global beat shape
        Conv1D(32→64, kernel=3) → ReLU → AdaptiveAvgPool(2) → shape (batch, 64, 2)

      Flatten → Linear → ReLU → Linear → (batch, latent_dim)

    AdaptiveAvgPool(2): compresses to exactly 2 time positions.
    Using 2 (not 8) because MPS requires input_size % output_size == 0.
    After two MaxPool1d(2) on 187 samples: floor(187/2)=93 → floor(93/2)=46.
    46 % 2 == 0 (MPS-safe), but 46 % 8 == 6 (crashes on MPS).
    """

    def __init__(self, input_len: int = 187, latent_dim: int = 32):
        super().__init__()

        self.latent_dim = latent_dim

        self.conv_blocks = nn.Sequential(
            # Block 1 — local patterns
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),                 # (batch, 16, 93)

            # Block 2 — beat structure
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),                 # (batch, 32, 46)

            # Block 3 — global shape
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(2),                     # (batch, 64, 2) — 46%2==0, MPS-safe
        )

        self.fc = nn.Sequential(
            nn.Flatten(),                                 # (batch, 128)
            nn.Linear(64 * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, latent_dim),                  # (batch, latent_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, 187) — raw ECG signal
        returns: (batch, latent_dim) — compact feature vector
        """
        x = x.unsqueeze(1)          # add channel dim → (batch, 1, 187)
        x = self.conv_blocks(x)
        x = self.fc(x)
        return x
