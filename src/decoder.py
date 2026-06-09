"""
src/decoder.py — MLP Decoder for autoencoder pretraining

Used only in Stage 1 of training to prevent encoder collapse.
Discarded after Stage 1 — not part of the saved model.

Why this is needed:
  Without a reconstruction objective, the CNN encoder learns the
  easiest way to maximise MAF log-likelihood: output near-zero vectors,
  which sit at the peak of N(0,I). Every input — normal or anomaly —
  gets the same features, so AUROC collapses to 0.5.

  Forcing the encoder to reconstruct the original 187-point signal means
  the 32/64 features must actually carry information about the waveform
  shape. Once the encoder is pretrained and frozen, the MAF learns density
  on genuinely informative features, and anomaly beats produce visibly
  different features at inference time.
"""

import torch.nn as nn


class ECGDecoder(nn.Module):
    """
    MLP that maps latent_dim features back to a 187-point ECG signal.
    Simple but effective — achieves lower MSE than CNN decoder in practice
    because it has fewer parameters to optimise and converges faster.
    """

    def __init__(self, latent_dim: int = 64, output_len: int = 187):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_len),
        )

    def forward(self, z):
        """z: (batch, latent_dim) → (batch, output_len)"""
        return self.fc(z)
