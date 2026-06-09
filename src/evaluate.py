"""
src/evaluate.py — Model evaluation for anomaly detection

Metrics used:
  AUROC  — Area Under ROC Curve. Main metric. 1.0 = perfect, 0.5 = random.
            Measures how well log p(x) separates normal from anomaly
            across ALL possible thresholds.

  Threshold — The specific log p(x) cutoff we use at deployment.
              Below threshold = anomaly. Chosen to maximise F1.

  Precision — Of beats we flagged as anomaly, how many actually were?
  Recall    — Of actual anomalies, how many did we catch?
  F1        — Harmonic mean of precision and recall.

Why AUROC is the primary metric:
  In medical settings you may want high recall (catch everything)
  or high precision (avoid false alarms). AUROC captures performance
  across the entire tradeoff curve, independent of threshold choice.
"""

import torch
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_fscore_support,
    confusion_matrix
)


def compute_scores(model, test_loader, device: str = "cpu") -> tuple:
    """
    Run all test beats through the model and collect log p(x) scores.

    Returns:
        scores : (N,) numpy array — log p(x) for each beat
                 higher = more normal, lower = more anomalous
        labels : (N,) numpy array — 0=normal, 1=anomaly (binary)
    """
    model.eval()
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            log_p = model.log_prob(x).cpu().numpy()
            all_scores.append(log_p)
            all_labels.append(y.numpy())

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)

    # Clip ±inf that appear when beats land far outside the learned boundary.
    # AUROC is unaffected (ranking-based), but threshold search needs finite values.
    scores = np.clip(scores, -1e6, 1e6)

    return scores, labels


def find_best_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Find the log p(x) threshold that maximises F1 score.

    Generates 500 candidate thresholds spanning the 1st–99th percentile
    of the actual score distribution, then picks the one with best F1.

    Why not use roc_curve thresholds directly:
      roc_curve(labels, -scores) returns thresholds in -scores space.
      Comparing those against scores (different space) silently breaks
      when the model is tight and anomaly scores approach -inf.
    """
    candidates = np.percentile(scores, np.linspace(1, 99, 500))

    best_f1     = 0.0
    best_thresh = candidates[0]

    for thresh in candidates:
        preds = (scores < thresh).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            labels, preds, average='binary', zero_division=0
        )
        if f1 > best_f1:
            best_f1     = f1
            best_thresh = thresh

    return float(best_thresh)


def evaluate(
    model,
    test_loader,
    device:   str = "cpu",
    save_dir: str = "outputs",
) -> dict:
    """
    Full evaluation pipeline. Returns metrics dict and saves results.
    """
    print("\nEvaluating on test set...")
    scores, labels = compute_scores(model, test_loader, device)

    # AUROC — primary metric
    auroc = roc_auc_score(labels, -scores)

    # Find best threshold
    threshold = find_best_threshold(scores, labels)

    # Apply threshold
    preds = (scores < threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary', zero_division=0
    )
    cm = confusion_matrix(labels, preds)

    tn, fp, fn, tp = cm.ravel()

    results = {
        "auroc":             round(float(auroc),     4),
        "threshold":         round(float(threshold), 4),
        "precision":         round(float(precision), 4),
        "recall":            round(float(recall),    4),
        "f1":                round(float(f1),        4),
        "true_positives":    int(tp),
        "false_positives":   int(fp),
        "true_negatives":    int(tn),
        "false_negatives":   int(fn),
        "total_test_beats":  int(len(labels)),
        "anomaly_beats":     int(labels.sum()),
        "normal_beats":      int((labels == 0).sum()),
    }

    # Print summary
    print(f"\n{'─'*40}")
    print(f"  AUROC           : {auroc:.4f}")
    print(f"  Threshold       : {threshold:.4f}")
    print(f"  Precision       : {precision:.4f}")
    print(f"  Recall          : {recall:.4f}")
    print(f"  F1 Score        : {f1:.4f}")
    print(f"{'─'*40}")
    print(f"  True  Positives : {tp:,}  (anomalies correctly caught)")
    print(f"  False Positives : {fp:,}  (normal beats wrongly flagged)")
    print(f"  True  Negatives : {tn:,}  (normal beats correctly passed)")
    print(f"  False Negatives : {fn:,}  (anomalies missed)")
    print(f"{'─'*40}")

    # Save results
    path = Path(save_dir) / "eval_results.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {path}")

    # Return everything needed for visualisation
    fpr, tpr, _ = roc_curve(labels, -scores)
    results["_scores"]    = scores
    results["_labels"]    = labels
    results["_fpr"]       = fpr
    results["_tpr"]       = tpr

    return results
