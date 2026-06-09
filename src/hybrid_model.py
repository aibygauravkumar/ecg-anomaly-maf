"""
src/hybrid_model.py — CNN Encoder + MAF hybrid anomaly detector

Architecture:
  Raw ECG (187) → CNN Encoder → features (latent_dim) → MAF → log p(features)

Why this works better than raw MAF:
  The CNN transforms a noisy 187-point waveform into a compact 32-number vector
  that captures only the clinically meaningful shape information. The MAF then
  estimates density on this clean, low-dimensional space instead of fighting
  through noise, baseline wander, and alignment jitter.

Training is end-to-end: MAF's log-likelihood gradient flows back through the
encoder, so the CNN learns to produce features that are maximally useful for
density estimation — not generic features, but anomaly-detection features.
"""

import torch
import torch.nn as nn
from src.encoder import ECGEncoder
from src.maf_model import MAF


class HybridECGModel(nn.Module):
    """
    ECGEncoder (CNN) + MAF stacked into one model.

    External interface is identical to the standalone MAF:
      model.log_prob(x)  — x is raw ECG (batch, 187), not pre-encoded features
    This means evaluate.py, detect.py, and the training loop need zero changes
    in how they call the model.
    """

    def __init__(
        self,
        input_len:   int       = 187,
        latent_dim:  int       = 32,
        n_layers:    int       = 5,
        hidden_dims: list[int] = [256, 256],
    ):
        super().__init__()
        self.encoder    = ECGEncoder(input_len=input_len, latent_dim=latent_dim)
        self.maf        = MAF(input_dim=latent_dim, n_layers=n_layers, hidden_dims=hidden_dims)
        self.latent_dim = latent_dim

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        x       : (batch, 187) — raw normalised ECG signal
        returns : (batch,)     — log p(encoder(x)) under the MAF density
        """
        features = self.encoder(x)          # (batch, latent_dim)
        return self.maf.log_prob(features)  # (batch,)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return CNN features without computing log-likelihood."""
        with torch.no_grad():
            return self.encoder(x)
