"""
ETAFlow Classification Decision Threshold Optimization Module.

Evaluates decision thresholds across the [0.20, 0.80] range on validation data
to find the optimal balance of Precision, Recall, and F1 score for delivery delay alerts.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)


def evaluate_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """Calculate classification performance metrics at a specific decision threshold.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_prob: Predicted positive-class probability.
        threshold: Decision boundary between 0 and 1.

    Returns:
        Dict[str, Any]: Metrics dictionary including precision, recall, f1, accuracy, FP, FN.
    """
    preds = (y_prob >= threshold).astype(int)

    acc = float(accuracy_score(y_true, preds))
    prec = float(precision_score(y_true, preds, zero_division=0))
    rec = float(recall_score(y_true, preds, zero_division=0))
    f1 = float(f1_score(y_true, preds, zero_division=0))

    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "threshold": round(threshold, 4),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold_range: Tuple[float, float] = (0.20, 0.80),
    step: float = 0.02,
    target_metric: str = "f1",
) -> Dict[str, Any]:
    """Sweep thresholds across a range on validation data and select the best operating point.

    Args:
        y_true: Ground truth binary labels.
        y_prob: Predicted positive-class probabilities.
        threshold_range: (min_thresh, max_thresh) tuple.
        step: Step size for threshold grid.
        target_metric: Objective metric to maximize (default: 'f1').

    Returns:
        Dict[str, Any]: Summary of search containing optimal threshold, best metrics,
        baseline 0.5 metrics, and the full threshold curve.
    """
    min_t, max_t = threshold_range
    thresholds = np.arange(min_t, max_t + step / 2, step)

    records: List[Dict[str, Any]] = []
    best_record: Optional[Dict[str, Any]] = None
    best_score = -1.0

    for t in thresholds:
        rec = evaluate_threshold(y_true, y_prob, float(t))
        records.append(rec)

        score = rec.get(target_metric, 0.0)
        if score > best_score:
            best_score = score
            best_record = rec

    # Also extract default 0.50 baseline for comparison
    baseline_record = evaluate_threshold(y_true, y_prob, 0.50)

    logger.info(
        "Threshold optimization completed. Default 0.5 F1=%.4f -> Optimal %.2f F1=%.4f (Prec=%.4f, Rec=%.4f)",
        baseline_record["f1"],
        best_record["threshold"] if best_record else 0.5,
        best_record["f1"] if best_record else baseline_record["f1"],
        best_record["precision"] if best_record else baseline_record["precision"],
        best_record["recall"] if best_record else baseline_record["recall"],
    )

    return {
        "target_metric": target_metric,
        "optimal_threshold": best_record["threshold"] if best_record else 0.5,
        "optimal_metrics": best_record,
        "baseline_threshold": 0.50,
        "baseline_metrics": baseline_record,
        "improvement_delta_f1": (
            (best_record["f1"] - baseline_record["f1"]) if best_record else 0.0
        ),
        "threshold_curve": records,
    }
