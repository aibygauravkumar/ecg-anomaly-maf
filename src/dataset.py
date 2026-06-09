"""
src/dataset.py — ECG data loading and preprocessing

Dataset: MIT-BIH Arrhythmia Database (preprocessed CSV version)
Source:  Kaggle — https://www.kaggle.com/datasets/shayanfazeli/heartbeat

Download instructions:
  1. Go to the Kaggle link above
  2. Download mitbih_train.csv and mitbih_test.csv
  3. Place both files in the data/ folder

Format:
  Each row = one heartbeat = 187 values + 1 label
  Columns 0-186: ECG signal (187 time steps)
  Column  187:   label
    0 = Normal
    1 = Supraventricular (atrial) arrhythmia
    2 = Ventricular arrhythmia
    3 = Fusion beat
    4 = Unclassifiable

Anomaly detection strategy:
  Train ONLY on label=0 (Normal) heartbeats
  At inference: low log p(x) = likely anomaly
  We never show the model what anomalies look like during training
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

INPUT_DIM = 187    # 187 time steps per heartbeat

CLASS_NAMES = {
    0: "Normal",
    1: "Supraventricular",
    2: "Ventricular",
    3: "Fusion",
    4: "Unclassifiable"
}


def load_raw(data_dir: str = "data") -> tuple:
    """
    Load raw CSV files and return (features, labels) for train and test.
    Raises clear error if files are not found.
    """
    train_path = Path(data_dir) / "mitbih_train.csv"
    test_path  = Path(data_dir) / "mitbih_test.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "\nDataset not found. Please:\n"
            "  1. Go to https://www.kaggle.com/datasets/shayanfazeli/heartbeat\n"
            "  2. Download mitbih_train.csv and mitbih_test.csv\n"
            "  3. Place both files in the data/ folder\n"
        )

    train_df = pd.read_csv(train_path, header=None)
    test_df  = pd.read_csv(test_path,  header=None)

    X_train = train_df.iloc[:, :187].values.astype(np.float32)
    y_train = train_df.iloc[:,  187].values.astype(int)

    X_test  = test_df.iloc[:,  :187].values.astype(np.float32)
    y_test  = test_df.iloc[:,   187].values.astype(int)

    return X_train, y_train, X_test, y_test


def get_dataloaders(
    data_dir:   str   = "data",
    batch_size: int   = 256,
    val_split:  float = 0.1,
) -> tuple:
    """
    Returns train_loader, val_loader, test_loader, mean, std.

    Training and validation use NORMAL beats only (label=0).
    Test loader contains all classes for evaluation.

    Why train on normal only?
      Anomaly detection is an unsupervised problem.
      We teach the model what normal looks like.
      Anything with low p(x) is flagged as unusual.
    """
    X_train, y_train, X_test, y_test = load_raw(data_dir)

    # Keep only normal beats for training
    normal_mask = y_train == 0
    X_normal    = X_train[normal_mask]

    print(f"Total training beats : {len(X_train):,}")
    print(f"Normal beats (train) : {len(X_normal):,}  ({len(X_normal)/len(X_train)*100:.1f}%)")
    print(f"Test beats           : {len(X_test):,}")

    # Compute normalisation statistics from normal training beats only
    mean = X_normal.mean(axis=0)
    std  = X_normal.std(axis=0)
    std  = np.where(std < 1e-6, 1.0, std)   # avoid division by zero

    # Normalise
    X_normal_norm = (X_normal - mean) / std
    X_test_norm   = (X_test   - mean) / std

    # Convert to tensors
    mean_t = torch.tensor(mean, dtype=torch.float32)
    std_t  = torch.tensor(std,  dtype=torch.float32)
    X_n    = torch.tensor(X_normal_norm, dtype=torch.float32)
    X_te   = torch.tensor(X_test_norm,   dtype=torch.float32)
    y_te   = torch.tensor(y_test,        dtype=torch.long)

    # Binary labels for test: 0=normal, 1=anomaly
    y_binary = (y_te > 0).long()

    # Train / val split on normal beats
    n_val   = int(len(X_n) * val_split)
    n_train = len(X_n) - n_val
    indices = torch.randperm(len(X_n), generator=torch.Generator().manual_seed(42))
    train_idx = indices[:n_train]
    val_idx   = indices[n_train:]

    train_loader = DataLoader(
        TensorDataset(X_n[train_idx]),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_n[val_idx]),
        batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(X_te, y_binary),
        batch_size=batch_size, shuffle=False
    )

    return train_loader, val_loader, test_loader, mean_t, std_t


def get_sample_beats(data_dir: str = "data", n_per_class: int = 5) -> dict:
    """
    Returns a few example beats per class for visualisation.
    Used by visualize.py to show what each type of heartbeat looks like.
    """
    X_train, y_train, _, _ = load_raw(data_dir)
    samples = {}
    for cls in range(5):
        mask = y_train == cls
        if mask.sum() > 0:
            samples[cls] = X_train[mask][:n_per_class]
    return samples


if __name__ == "__main__":
    train_loader, val_loader, test_loader, mean, std = get_dataloaders()
    batch = next(iter(train_loader))[0]
    print(f"\nBatch shape : {batch.shape}")
    print(f"Batch mean  : {batch.mean():.4f}")
    print(f"Batch std   : {batch.std():.4f}")
    print("\ndataset.py working correctly.")
