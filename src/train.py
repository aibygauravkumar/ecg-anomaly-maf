"""
src/train.py — Two-stage training for CNN + MAF anomaly detection

Stage 1 — Encoder pretraining (autoencoder):
  Train CNN encoder + MLP decoder with MSE reconstruction loss.
  This forces the 32 latent features to carry real information about
  the waveform. Without this, the encoder collapses to outputting
  near-zero vectors (which maximise MAF log-prob trivially), causing
  AUROC to drop to 0.5.

Stage 2 — MAF training (frozen encoder):
  Freeze the encoder. Train only the MAF to model p(features) for
  normal beats. At inference, anomaly beats produce out-of-distribution
  features → low log p → flagged as anomaly.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from src.dataset      import get_dataloaders, INPUT_DIM
from src.hybrid_model import HybridECGModel
from src.decoder      import ECGDecoder


def train(
    data_dir:        str   = "data",
    latent_dim:      int   = 32,
    n_layers:        int   = 5,
    hidden_dims:     list  = [256, 256],
    pretrain_epochs: int   = 30,
    n_epochs:        int   = 100,
    lr:              float = 1e-4,
    weight_decay:    float = 1e-4,
    batch_size:      int   = 256,
    patience:        int   = 15,
    device:          str   = "cpu",
    save_dir:        str   = "outputs",
) -> tuple:

    Path(save_dir).mkdir(exist_ok=True)

    # ── Data ────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, mean, std = get_dataloaders(
        data_dir=data_dir, batch_size=batch_size
    )

    # ── Model ────────────────────────────────────────────────────────────
    model = HybridECGModel(
        input_len=INPUT_DIM,
        latent_dim=latent_dim,
        n_layers=n_layers,
        hidden_dims=hidden_dims,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel      : CNN ({INPUT_DIM}→{latent_dim}) + MAF ({n_layers} layers, hidden={hidden_dims})")
    print(f"Parameters : {n_params:,}")
    print(f"Device     : {device.upper()}")

    # ════════════════════════════════════════════════════════════════════
    # STAGE 1 — Autoencoder pretraining
    # Encoder + Decoder trained with MSE loss.
    # Goal: force encoder to produce features that carry real waveform info.
    # ════════════════════════════════════════════════════════════════════
    decoder = ECGDecoder(latent_dim=latent_dim, output_len=INPUT_DIM).to(device)
    ae_optimizer = optim.Adam(
        list(model.encoder.parameters()) + list(decoder.parameters()),
        lr=lr, weight_decay=weight_decay
    )

    print(f"\n── Stage 1: Encoder pretraining ({pretrain_epochs} epochs, MSE loss) ──")
    print(f"{'Epoch':>6} | {'Train MSE':>10} | {'Val MSE':>10}")
    print("-" * 32)

    for epoch in range(1, pretrain_epochs + 1):
        model.encoder.train()
        decoder.train()
        train_mse = 0.0
        for (x,) in train_loader:
            x = x.to(device)
            ae_optimizer.zero_grad()
            z    = model.encoder(x)
            x_hat = decoder(z)
            loss = nn.functional.mse_loss(x_hat, x)
            loss.backward()
            ae_optimizer.step()
            train_mse += loss.item() * len(x)
        train_mse /= len(train_loader.dataset)

        model.encoder.eval()
        decoder.eval()
        val_mse = 0.0
        with torch.no_grad():
            for (x,) in val_loader:
                x = x.to(device)
                z     = model.encoder(x)
                x_hat = decoder(z)
                val_mse += nn.functional.mse_loss(x_hat, x).item() * len(x)
        val_mse /= len(val_loader.dataset)

        if epoch % 5 == 0 or epoch == 1:
            print(f"{epoch:>6} | {train_mse:>10.4f} | {val_mse:>10.4f}")

    print("Encoder pretraining done. Freezing encoder weights.")

    # Freeze encoder — MAF training will not update these parameters
    for param in model.encoder.parameters():
        param.requires_grad = False

    # ════════════════════════════════════════════════════════════════════
    # STAGE 2 — MAF training (frozen encoder)
    # Only MAF parameters are updated. The encoder is fixed.
    # ════════════════════════════════════════════════════════════════════
    maf_optimizer = optim.Adam(
        model.maf.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        maf_optimizer, mode='min', factor=0.5, patience=5
    )

    print(f"\n── Stage 2: MAF training ({n_epochs} epochs, NLL loss) ──")
    print(f"{'Epoch':>6} | {'Train NLL':>10} | {'Val NLL':>10} | {'LR':>10}")
    print("-" * 45)

    train_losses, val_losses = [], []
    best_val   = float('inf')
    no_improve = 0
    best_path  = Path(save_dir) / "best_model.pt"

    for epoch in range(1, n_epochs + 1):

        model.train()
        model.encoder.eval()   # keep BN running stats frozen during MAF training
        train_nll = 0.0
        for (x,) in train_loader:
            x = x.to(device)
            maf_optimizer.zero_grad()
            loss = -model.log_prob(x).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.maf.parameters(), max_norm=1.0)
            maf_optimizer.step()
            train_nll += loss.item() * len(x)
        train_nll /= len(train_loader.dataset)

        model.eval()
        val_nll = 0.0
        with torch.no_grad():
            for (x,) in val_loader:
                x = x.to(device)
                val_nll += (-model.log_prob(x).mean()).item() * len(x)
        val_nll /= len(val_loader.dataset)

        scheduler.step(val_nll)
        train_losses.append(train_nll)
        val_losses.append(val_nll)

        if val_nll < best_val:
            best_val   = val_nll
            no_improve = 0
            torch.save(model.state_dict(), best_path)
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            lr_now = maf_optimizer.param_groups[0]['lr']
            print(f"{epoch:>6} | {train_nll:>10.2f} | {val_nll:>10.2f} | {lr_now:>10.2e}")

        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    print(f"\nBest val NLL : {best_val:.2f}")
    print(f"Saved to     : {best_path}")

    _plot_losses(train_losses, val_losses, save_dir)

    config = dict(
        latent_dim=latent_dim,
        n_layers=n_layers, hidden_dims=hidden_dims,
        pretrain_epochs=pretrain_epochs,
        n_epochs=n_epochs, lr=lr, weight_decay=weight_decay,
        batch_size=batch_size, device=device,
        best_val_nll=best_val
    )
    with open(Path(save_dir) / "train_config.json", "w") as f:
        json.dump(config, f, indent=2)

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    return model, mean, std, test_loader


def _plot_losses(train_losses, val_losses, save_dir):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label='Train NLL', color='steelblue')
    ax.plot(val_losses,   label='Val NLL',   color='tomato')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Negative Log-Likelihood  (Stage 2 — MAF only)")
    ax.set_title("CNN + MAF Training — ECG Anomaly Detection")
    ax.legend()
    plt.tight_layout()
    path = Path(save_dir) / "training_curves.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves → {path}")
