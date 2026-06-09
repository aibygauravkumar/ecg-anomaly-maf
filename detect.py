"""
detect.py — Interactive ECG anomaly detector

After training (python main.py), run:
  python detect.py

You will be shown real heartbeats from the test set one at a time.
The model scores each beat and tells you whether it is normal or anomalous.
Press Enter to see the next beat. Type 'quit' to exit.

This simulates a real deployment scenario:
  A new ECG reading arrives → model scores it → alert if anomalous
"""

import torch
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from src.dataset      import load_raw, INPUT_DIM, CLASS_NAMES
from src.hybrid_model import HybridECGModel


def load_artifacts(save_dir: str = "outputs") -> tuple:
    """Load trained model and evaluation threshold."""
    model_path  = Path(save_dir) / "best_model.pt"
    config_path = Path(save_dir) / "train_config.json"
    eval_path   = Path(save_dir) / "eval_results.json"

    for p in [model_path, config_path, eval_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Run 'python main.py' first."
            )

    with open(config_path) as f:
        config = json.load(f)
    with open(eval_path) as f:
        results = json.load(f)

    model = HybridECGModel(
        input_len=INPUT_DIM,
        latent_dim=config.get("latent_dim", 32),
        n_layers=config["n_layers"],
        hidden_dims=config["hidden_dims"],
    )
    model.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True)
    )
    model.eval()

    threshold = results["threshold"]
    auroc     = results["auroc"]
    return model, threshold, auroc


def score_beat(model, beat: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
    """Normalise one beat and compute its log p(x) score."""
    beat_norm = (beat - mean) / np.where(std < 1e-6, 1.0, std)
    x = torch.tensor(beat_norm, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        return model.log_prob(x).item()


def plot_beat(beat: np.ndarray, score: float, threshold: float,
              true_label: int, save_path: str):
    """Plot a single heartbeat with its score and verdict."""
    is_anomaly = score < threshold
    color      = "tomato" if is_anomaly else "steelblue"
    verdict    = "ANOMALY" if is_anomaly else "NORMAL"

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(beat, color=color, linewidth=2)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Amplitude")
    ax.set_title(
        f"Verdict: {verdict}  |  Score: {score:.2f}  |  "
        f"Threshold: {threshold:.2f}  |  True label: {CLASS_NAMES[true_label]}",
        fontsize=11, color=color
    )
    ax.set_xlim(0, len(beat) - 1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()


def run(save_dir: str = "outputs"):
    """Interactive detection loop."""
    print("\nLoading model...")
    model, threshold, auroc = load_artifacts(save_dir)

    print("Loading test data...")
    X_train, y_train, X_test, y_test = load_raw()

    # Compute mean/std from normal training beats (same as training)
    normal_mask = y_train == 0
    mean = X_train[normal_mask].mean(axis=0)
    std  = X_train[normal_mask].std(axis=0)

    Path(save_dir).mkdir(exist_ok=True)
    output_dir = Path(save_dir) / "detections"
    output_dir.mkdir(exist_ok=True)

    print(f"\n{'='*50}")
    print(f"  ECG Anomaly Detector — Interactive Mode")
    print(f"{'='*50}")
    print(f"  Model AUROC : {auroc:.4f}")
    print(f"  Threshold   : {threshold:.4f}")
    print(f"  Score < threshold → ANOMALY")
    print(f"  Score > threshold → NORMAL")
    print(f"\n  Press Enter for next beat. Type 'quit' to exit.\n")

    # Shuffle test set for variety
    rng = np.random.default_rng(seed=0)
    indices = rng.permutation(len(X_test))
    count = 0

    for idx in indices:
        beat       = X_test[idx]
        true_label = y_test[idx]
        score      = score_beat(model, beat, mean, std)
        is_anomaly = score < threshold
        verdict    = "ANOMALY ⚠️ " if is_anomaly else "NORMAL  ✓ "
        correct    = (is_anomaly == (true_label > 0))

        print(f"Beat #{count+1:04d} | Score: {score:8.2f} | {verdict} | "
              f"True: {CLASS_NAMES[true_label]:<18} | {'✓ Correct' if correct else '✗ Wrong'}")

        # Save plot
        save_path = str(output_dir / f"beat_{count+1:04d}.png")
        plot_beat(beat, score, threshold, true_label, save_path)

        count += 1
        try:
            cmd = input("  [Enter] next  |  [q] quit > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        if cmd in ("q", "quit", "exit"):
            break

    print(f"\nDetection session ended. {count} beats reviewed.")
    print(f"Beat plots saved to {output_dir}/")


if __name__ == "__main__":
    run()
