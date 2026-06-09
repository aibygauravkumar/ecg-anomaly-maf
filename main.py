"""
main.py — Train and evaluate MAF for ECG anomaly detection

Usage:
  python main.py                   # full train + evaluate
  python main.py --eval-only       # skip training, re-evaluate saved model
  python main.py --device mps      # Mac GPU
  python main.py --device cuda     # NVIDIA GPU
  python main.py --device cpu      # CPU

Outputs saved to outputs/:
  best_model.pt          — trained model weights
  train_config.json      — hyperparameters used
  eval_results.json      — AUROC, F1, precision, recall
  training_curves.png    — NLL over epochs
  ecg_samples.png        — example beats per class
  score_distribution.png — how model separates normal vs anomaly
  roc_curve.png          — ROC curve with AUROC
"""

import argparse
import json
import torch
from pathlib import Path


def get_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main(device: str = "auto", eval_only: bool = False, train_cnn: bool = False):
    device = get_device(device)

    print("=" * 50)
    print("  ECG Anomaly Detection — CNN + MAF")
    print("=" * 50)
    print(f"  Device : {device.upper()}")

    from src.dataset   import get_sample_beats, get_dataloaders
    from src.evaluate  import evaluate
    from src.visualize import plot_ecg_samples, plot_score_distribution, plot_roc_curve

    # ── Step 1: Visualise the raw ECG data ──────────────────────────────
    print("\n[1/4] Plotting ECG sample beats...")
    plot_ecg_samples(get_sample_beats(), save_dir="outputs")

    if eval_only:
        # ── Load saved model and config ──────────────────────────────────
        print("\n[2/4] Loading saved model (--eval-only, skipping training)...")
        from src.hybrid_model import HybridECGModel
        from src.dataset      import INPUT_DIM

        config_path = Path("outputs/train_config.json")
        model_path  = Path("outputs/best_model.pt")
        if not config_path.exists() or not model_path.exists():
            raise FileNotFoundError("No saved model found. Run without --eval-only first.")

        with open(config_path) as f:
            config = json.load(f)

        model = HybridECGModel(
            input_len   = INPUT_DIM,
            latent_dim  = config.get("latent_dim", 64),
            n_layers    = config["n_layers"],
            hidden_dims = config["hidden_dims"],
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

        _, _, test_loader, mean, std = get_dataloaders(data_dir="data", batch_size=256)

    else:
        # ── Step 2: Train ────────────────────────────────────────────────
        from src.train import train
        print("\n[2/4] Training CNN + MAF on normal heartbeats only...")
        model, mean, std, test_loader = train(
            data_dir        = "data",
            latent_dim      = 64,
            n_layers        = 8,
            hidden_dims     = [512, 512],
            pretrain_epochs = 100,
            n_epochs        = 400,
            lr              = 1e-4,
            weight_decay    = 1e-4,
            batch_size      = 256,
            patience        = 15,
            device          = device,
            save_dir        = "outputs",
        )

    # ── Step 3: Evaluate ─────────────────────────────────────────────────
    print("\n[3/4] Evaluating on test set (normal + anomaly beats)...")
    results = evaluate(model, test_loader, device=device, save_dir="outputs")

    # ── Step 4: Visualise results ─────────────────────────────────────────
    print("\n[4/4] Saving result plots...")
    plot_score_distribution(
        results["_scores"], results["_labels"],
        results["threshold"], save_dir="outputs"
    )
    plot_roc_curve(
        results["_fpr"], results["_tpr"],
        results["auroc"], save_dir="outputs"
    )

    print("\n" + "=" * 50)
    print(f"  AUROC  : {results['auroc']:.4f}")
    print(f"  F1     : {results['f1']:.4f}")
    print(f"  Recall : {results['recall']:.4f}")
    print("=" * 50)
    print("\nDone. Check outputs/ for all results.")

    if train_cnn:
        print("\n" + "=" * 50)
        print("  Training CNN classifier for comparison demo")
        print("=" * 50)
        from src.dataset        import load_raw
        from src.cnn_classifier import train_cnn as run_cnn
        X_train, y_train, _, _ = load_raw()
        run_cnn(X_train, y_train, save_dir="outputs", n_epochs=30, device=device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ECG Anomaly Detection with CNN + MAF")
    parser.add_argument(
        "--device", default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Device to use (default: auto-detect)"
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Skip training — load saved model from outputs/ and re-evaluate"
    )
    parser.add_argument(
        "--train-cnn", action="store_true",
        help="Also train the supervised CNN classifier (for the vs-classifier demo in app.py)"
    )
    args = parser.parse_args()
    main(device=args.device, eval_only=args.eval_only, train_cnn=args.train_cnn)
