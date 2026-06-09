"""
src/visualize.py — All plots for ECG anomaly detection

Produces four output images:
  1. ecg_samples.png        — what each heartbeat type looks like
  2. score_distribution.png — how model scores separate normal vs anomaly
  3. roc_curve.png          — AUROC visualisation
  4. training_curves.png    — NLL over epochs (produced by train.py)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from src.dataset import CLASS_NAMES


# Consistent colors per class across all plots
CLASS_COLORS = {
    0: "steelblue",
    1: "tomato",
    2: "darkorange",
    3: "green",
    4: "purple"
}


def plot_ecg_samples(samples: dict, save_dir: str = "outputs"):
    """
    Plot one example heartbeat per class so the viewer understands
    what each type of ECG signal looks like.

    Normal beats have a clear PQRST wave shape.
    Arrhythmias deviate from this pattern in characteristic ways.
    """
    n_classes = len(samples)
    fig, axes = plt.subplots(n_classes, 1, figsize=(10, n_classes * 2))
    fig.suptitle("ECG Heartbeat Types — MIT-BIH Dataset", fontsize=14, y=1.01)

    for cls, ax in zip(sorted(samples.keys()), axes):
        beat = samples[cls][0]       # first example of this class
        ax.plot(beat, color=CLASS_COLORS[cls], linewidth=1.5)
        ax.set_title(f"Class {cls}: {CLASS_NAMES[cls]}", fontsize=11)
        ax.set_xlabel("Time step")
        ax.set_ylabel("Amplitude")
        ax.set_xlim(0, len(beat) - 1)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = Path(save_dir) / "ecg_samples.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"ECG samples    → {path}")


def plot_score_distribution(
    scores:    np.ndarray,
    labels:    np.ndarray,
    threshold: float,
    save_dir:  str = "outputs"
):
    """
    Histogram of log p(x) scores split by normal vs anomaly.

    A good model produces two well-separated distributions:
      Normal beats:   high log p(x)  — model says "yes, this is familiar"
      Anomaly beats:  low  log p(x)  — model says "I have not seen this before"

    The threshold is the vertical line where we draw the boundary.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    normal_scores  = scores[labels == 0]
    anomaly_scores = scores[labels == 1]

    ax.hist(normal_scores,  bins=80, alpha=0.6,
            color='steelblue', label='Normal beats',  density=True)
    ax.hist(anomaly_scores, bins=80, alpha=0.6,
            color='tomato',    label='Anomaly beats', density=True)

    ax.axvline(threshold, color='black', linewidth=2,
               linestyle='--', label=f'Threshold = {threshold:.2f}')

    ax.set_xlabel("log p(x)  — Model score (higher = more normal)")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution: Normal vs Anomaly")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = Path(save_dir) / "score_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Score dist     → {path}")


def plot_roc_curve(
    fpr:     np.ndarray,
    tpr:     np.ndarray,
    auroc:   float,
    save_dir: str = "outputs"
):
    """
    ROC Curve — shows the tradeoff between catching anomalies
    and raising false alarms at every possible threshold.

    Perfect model: curve goes straight up then right (AUROC = 1.0)
    Random model:  diagonal line (AUROC = 0.5)

    Medical context:
      Moving threshold left  → catch more anomalies (high recall) but more false alarms
      Moving threshold right → fewer false alarms but miss more anomalies
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot(fpr, tpr, color='steelblue', linewidth=2,
            label=f'CNN + MAF (AUROC = {auroc:.4f})')
    ax.plot([0, 1], [0, 1], color='grey', linewidth=1,
            linestyle='--', label='Random (AUROC = 0.50)')

    ax.set_xlabel("False Positive Rate  (normal beats wrongly flagged)")
    ax.set_ylabel("True Positive Rate  (anomalies correctly caught)")
    ax.set_title("ROC Curve — ECG Anomaly Detection")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = Path(save_dir) / "roc_curve.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"ROC curve      → {path}")


def plot_all(samples: dict, eval_results: dict, save_dir: str = "outputs"):
    """Convenience wrapper — generates all plots in one call."""
    plot_ecg_samples(samples, save_dir)
    plot_score_distribution(
        eval_results["_scores"],
        eval_results["_labels"],
        eval_results["threshold"],
        save_dir
    )
    plot_roc_curve(
        eval_results["_fpr"],
        eval_results["_tpr"],
        eval_results["auroc"],
        save_dir
    )
