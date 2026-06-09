"""
src/cnn_classifier.py — 1D CNN trained on KNOWN arrhythmia types only

This CNN represents the "standard supervised approach".
It is trained on:
  Class 0: Normal
  Class 1: Supraventricular arrhythmia
  Class 2: Ventricular arrhythmia
  Class 3: Fusion beat

It deliberately does NOT see Class 4 (Unclassifiable) during training.
Class 4 represents a "newly discovered" or "rare" condition.

This lets us demonstrate the core limitation of supervised classifiers:
they can only detect what they were explicitly trained to recognise.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path


KNOWN_CLASSES  = [0, 1, 2, 3]      # CNN trained on these
UNKNOWN_CLASS  = 4                  # CNN never sees this — the "new condition"
CLASS_LABELS   = {
    0: "Normal",
    1: "Supraventricular",
    2: "Ventricular",
    3: "Fusion",
    4: "Unclassifiable (unknown to CNN)",
}


# ─────────────────────────────────────────────────────────────────────────────
# Architecture — simple 1D CNN for time series classification
# ─────────────────────────────────────────────────────────────────────────────

class ECG_CNN(nn.Module):
    """
    1D Convolutional Neural Network for ECG classification.

    Input: (batch, 187) — one heartbeat per sample
    Output: (batch, 4)  — probability for each known class

    Architecture:
      Conv1D → ReLU → MaxPool  (learns local patterns like P,Q,R,S,T waves)
      Conv1D → ReLU → MaxPool  (learns higher-level beat structure)
      Flatten → Dense → Dense → 4-class output

    Why 1D convolution?
      ECG is a time series — the same local patterns (spikes, dips) appear
      at different time positions across different beats. Conv1D learns these
      patterns and can find them wherever they appear.
    """

    def __init__(self, input_len: int = 187, n_classes: int = 4):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),   # (batch, 32, 187)
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),                   # (batch, 32, 93)

            nn.Conv1d(32, 64, kernel_size=5, padding=2),  # (batch, 64, 93)
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),                   # (batch, 64, 46)

            nn.Conv1d(64, 128, kernel_size=3, padding=1), # (batch, 128, 46)
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),                   # (batch, 128, 23)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 23, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 187) → add channel dim → (batch, 1, 187)
        x = x.unsqueeze(1)
        x = self.features(x)
        x = self.classifier(x)
        return x                    # raw logits, (batch, n_classes)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_cnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    save_dir: str = "outputs",
    n_epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cpu",
) -> ECG_CNN:
    """
    Train CNN on known classes only (0, 1, 2, 3).
    Class 4 is excluded — it represents the unknown condition.
    """
    Path(save_dir).mkdir(exist_ok=True)

    # ── Filter to known classes only ─────────────────────────────────────
    mask    = np.isin(y_train, KNOWN_CLASSES)
    X_known = X_train[mask]
    y_known = y_train[mask]

    # Remap labels to 0..3 (required for CrossEntropyLoss)
    y_remapped = np.zeros_like(y_known)
    for new_idx, old_class in enumerate(KNOWN_CLASSES):
        y_remapped[y_known == old_class] = new_idx

    # Normalise
    mean = X_known.mean(axis=0)
    std  = X_known.std(axis=0)
    std  = np.where(std < 1e-6, 1.0, std)
    X_norm = (X_known - mean) / std

    # Save normalisation stats (needed at inference time)
    np.save(Path(save_dir) / "cnn_mean.npy", mean)
    np.save(Path(save_dir) / "cnn_std.npy",  std)

    X_t = torch.tensor(X_norm,     dtype=torch.float32)
    y_t = torch.tensor(y_remapped, dtype=torch.long)

    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)

    # ── Model ────────────────────────────────────────────────────────────
    model     = ECG_CNN(input_len=187, n_classes=len(KNOWN_CLASSES)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nCNN: {n_params:,} parameters, training on classes {KNOWN_CLASSES}")
    print(f"Training beats: {len(X_known):,}  (class 4 excluded — the unknown condition)")
    print(f"{'Epoch':>6} | {'Loss':>8} | {'Accuracy':>10}")
    print("-" * 30)

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(xb)
            correct    += (logits.argmax(1) == yb).sum().item()
            total      += len(xb)

        if epoch % 5 == 0 or epoch == 1:
            print(f"{epoch:>6} | {total_loss/total:>8.4f} | {correct/total*100:>9.2f}%")

    save_path = Path(save_dir) / "cnn_model.pt"
    torch.save(model.state_dict(), save_path)
    print(f"\nCNN saved → {save_path}")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def load_cnn(save_dir: str = "outputs") -> tuple:
    """Load trained CNN and normalisation stats."""
    model_path = Path(save_dir) / "cnn_model.pt"
    mean_path  = Path(save_dir) / "cnn_mean.npy"
    std_path   = Path(save_dir) / "cnn_std.npy"

    if not all(p.exists() for p in [model_path, mean_path, std_path]):
        return None, None, None

    model = ECG_CNN(input_len=187, n_classes=4)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    mean = np.load(mean_path)
    std  = np.load(std_path)
    return model, mean, std


def cnn_predict(
    model:  ECG_CNN,
    beat:   np.ndarray,
    mean:   np.ndarray,
    std:    np.ndarray,
) -> dict:
    """
    Predict class probabilities for one beat.

    Returns dict with:
      predicted_class   : int (0-3, index into KNOWN_CLASSES)
      predicted_label   : str
      probabilities     : array of 4 class probabilities
      confidence        : float, max probability
      says_normal       : bool
    """
    std_safe = np.where(std < 1e-6, 1.0, std)
    x_norm   = (beat - mean) / std_safe
    x_t      = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        logits = model(x_t)
        probs  = torch.softmax(logits, dim=1).squeeze().numpy()

    pred_idx   = int(probs.argmax())
    pred_class = KNOWN_CLASSES[pred_idx]

    return {
        "predicted_class": pred_class,
        "predicted_label": CLASS_LABELS[pred_class],
        "probabilities":   probs,
        "confidence":      float(probs.max()),
        "says_normal":     pred_class == 0,
    }
