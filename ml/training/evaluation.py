"""
ETAFlow ML Evaluation Metrics Module.

Computes comprehensive, production-grade regression and classification metrics
for delivery ETA and delay prediction benchmarks.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)

logger = logging.getLogger(__name__)


def evaluate_regression(
    y_true: Union[np.ndarray, List[float]],
    y_pred: Union[np.ndarray, List[float]],
    tolerances_days: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Calculate comprehensive delivery-time regression evaluation metrics.

    Args:
        y_true: Ground truth target values (actual_delivery_days).
        y_pred: Predicted target values.
        tolerances_days: Tolerances in days for SLA accuracy (default: [0.5, 1.0, 2.0]).

    Returns:
        Dict[str, Any]: Dictionary containing MAE, RMSE, R2, tolerance accuracies,
        and residual error distribution statistics.
    """
    if tolerances_days is None:
        tolerances_days = [0.5, 1.0, 2.0]

    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)

    if len(y_t) != len(y_p):
        raise ValueError(f"Shape mismatch: y_true ({len(y_t)}) vs y_pred ({len(y_p)})")
    if len(y_t) == 0:
        raise ValueError("Cannot evaluate empty predictions.")

    abs_errors = np.abs(y_t - y_p)
    residuals = y_p - y_t

    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(root_mean_squared_error(y_t, y_p))
    r2 = float(r2_score(y_t, y_p))

    within_tolerances = {}
    for tol in tolerances_days:
        key = f"within_{tol}_day" if tol != 1.0 else "within_1_day"
        within_tolerances[key] = float(np.mean(abs_errors <= tol))

    metrics: Dict[str, Any] = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        **within_tolerances,
        "error_statistics": {
            "median_abs_error": float(np.median(abs_errors)),
            "max_abs_error": float(np.max(abs_errors)),
            "p95_abs_error": float(np.percentile(abs_errors, 95)),
            "residual_mean": float(np.mean(residuals)),
            "residual_std": float(np.std(residuals)),
        },
        "sample_count": len(y_t),
    }
    return metrics


def evaluate_classification(
    y_true: Union[np.ndarray, List[int]],
    y_pred: Union[np.ndarray, List[int]],
    y_prob: Optional[Union[np.ndarray, List[float]]] = None,
) -> Dict[str, Any]:
    """Calculate comprehensive binary delay classification evaluation metrics.

    Args:
        y_true: Ground truth binary targets (is_delayed in {0, 1}).
        y_pred: Predicted discrete class labels (0 or 1).
        y_prob: Predicted probabilities for positive class (delay).

    Returns:
        Dict[str, Any]: Dictionary containing Accuracy, Precision, Recall, F1,
        ROC-AUC, PR-AUC, and Confusion Matrix breakdown.
    """
    y_t = np.asarray(y_true, dtype=np.int64)
    y_p = np.asarray(y_pred, dtype=np.int64)

    if len(y_t) != len(y_p):
        raise ValueError(f"Shape mismatch: y_true ({len(y_t)}) vs y_pred ({len(y_p)})")
    if len(y_t) == 0:
        raise ValueError("Cannot evaluate empty predictions.")

    acc = float(accuracy_score(y_t, y_p))
    prec = float(precision_score(y_t, y_p, zero_division=0))
    rec = float(recall_score(y_t, y_p, zero_division=0))
    f1 = float(f1_score(y_t, y_p, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_t, y_p, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # ROC-AUC & PR-AUC calculation with safe handling
    roc_auc: Optional[float] = None
    pr_auc: Optional[float] = None

    if y_prob is not None:
        y_pr = np.asarray(y_prob, dtype=np.float64)
        if len(y_pr) == len(y_t) and len(np.unique(y_t)) > 1:
            try:
                roc_auc = float(roc_auc_score(y_t, y_pr))
            except Exception as e:
                logger.warning("Failed to compute ROC-AUC: %s", e)
            try:
                pr_auc = float(average_precision_score(y_t, y_pr))
            except Exception as e:
                logger.warning("Failed to compute PR-AUC: %s", e)
    else:
        # If no probabilities, attempt ROC-AUC on discrete predictions
        if len(np.unique(y_t)) > 1:
            try:
                roc_auc = float(roc_auc_score(y_t, y_p))
                pr_auc = float(average_precision_score(y_t, y_p))
            except Exception:
                pass

    metrics: Dict[str, Any] = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "matrix": cm.tolist(),
        },
        "sample_count": len(y_t),
    }
    return metrics
