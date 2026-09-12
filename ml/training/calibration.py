"""
ETAFlow Probability Calibration Module.

Implements and benchmarks probability calibration methods (Platt Sigmoid and Isotonic)
to produce reliable, risk-reflective delay probabilities for operations.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from ml.training.classification import predict_classification_proba
from ml.training.evaluation import evaluate_classification

logger = logging.getLogger(__name__)


def evaluate_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Compute calibration curve and Brier score.

    Args:
        y_true: Ground truth binary labels.
        y_prob: Predicted probabilities.
        n_bins: Number of discretization bins.

    Returns:
        Dict[str, Any]: Fraction of positives, mean predicted value, Brier score, and bin details.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    brier = float(brier_score_loss(y_true, y_prob))

    # Expected Calibration Error (ECE) approximation
    bin_counts, _ = np.histogram(y_prob, bins=np.linspace(0, 1, n_bins + 1))
    valid_bins = bin_counts[: len(prob_true)] > 0
    weights = bin_counts[: len(prob_true)][valid_bins] / len(y_prob)
    ece = float(np.sum(weights * np.abs(prob_true[valid_bins] - prob_pred[valid_bins])))

    curve_points = [
        {"bin": idx + 1, "mean_pred": float(p_pred), "actual_pos_rate": float(p_true)}
        for idx, (p_pred, p_true) in enumerate(zip(prob_pred, prob_true))
    ]

    return {
        "brier_score": brier,
        "expected_calibration_error": ece,
        "calibration_curve": curve_points,
    }


def calibrate_and_evaluate(
    base_model: Any,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """Train, calibrate, and compare Sigmoid (Platt) and Isotonic post-hoc calibrators.

    Note: The base model is trained on X_train.
    Post-hoc calibrators are fitted on validation probabilities to prevent overoptimistic bias.

    Args:
        base_model: Fitted base classifier.
        X_train: Preprocessed train features.
        y_train: Train target array.
        X_val: Preprocessed validation features.
        y_val: Validation target array.
        threshold: Operating decision threshold.

    Returns:
        Dict[str, Any]: Comparison of uncalibrated, sigmoid, and isotonic performance.
    """
    # 1. Uncalibrated baseline on validation
    prob_uncal = predict_classification_proba(base_model, X_val)
    preds_uncal = (prob_uncal >= threshold).astype(int)
    metrics_uncal = evaluate_classification(y_val, preds_uncal, y_prob=prob_uncal)
    cal_uncal = evaluate_calibration_curve(y_val, prob_uncal)

    # 2. Sigmoid (Platt Scaling) Calibrator: Logistic regression on predicted probabilities
    logger.info("Fitting Sigmoid (Platt) calibrator on validation probabilities...")
    cal_sigmoid = LogisticRegression(solver="lbfgs", random_state=42)
    cal_sigmoid.fit(prob_uncal.reshape(-1, 1), y_val)
    prob_sigmoid = cal_sigmoid.predict_proba(prob_uncal.reshape(-1, 1))[:, 1]
    preds_sigmoid = (prob_sigmoid >= threshold).astype(int)
    metrics_sigmoid = evaluate_classification(y_val, preds_sigmoid, y_prob=prob_sigmoid)
    cal_curve_sigmoid = evaluate_calibration_curve(y_val, prob_sigmoid)

    # 3. Isotonic Calibrator: Non-parametric isotonic regression on predicted probabilities
    logger.info("Fitting Isotonic calibrator on validation probabilities...")
    cal_isotonic = IsotonicRegression(out_of_bounds="clip")
    cal_isotonic.fit(prob_uncal, y_val)
    prob_isotonic = cal_isotonic.predict(prob_uncal)
    preds_isotonic = (prob_isotonic >= threshold).astype(int)
    metrics_isotonic = evaluate_classification(y_val, preds_isotonic, y_prob=prob_isotonic)
    cal_curve_isotonic = evaluate_calibration_curve(y_val, prob_isotonic)

    results = {
        "uncalibrated": {
            "model_type": "Uncalibrated",
            "brier_score": cal_uncal["brier_score"],
            "ece": cal_uncal["expected_calibration_error"],
            "f1": metrics_uncal["f1"],
            "precision": metrics_uncal["precision"],
            "recall": metrics_uncal["recall"],
            "roc_auc": metrics_uncal.get("roc_auc"),
            "pr_auc": metrics_uncal.get("pr_auc"),
            "calibration_curve": cal_uncal["calibration_curve"],
        },
        "sigmoid": {
            "model_type": "Platt Sigmoid",
            "brier_score": cal_curve_sigmoid["brier_score"],
            "ece": cal_curve_sigmoid["expected_calibration_error"],
            "f1": metrics_sigmoid["f1"],
            "precision": metrics_sigmoid["precision"],
            "recall": metrics_sigmoid["recall"],
            "roc_auc": metrics_sigmoid.get("roc_auc"),
            "pr_auc": metrics_sigmoid.get("pr_auc"),
            "calibration_curve": cal_curve_sigmoid["calibration_curve"],
            "calibrator": cal_sigmoid,
        },
        "isotonic": {
            "model_type": "Isotonic Regression",
            "brier_score": cal_curve_isotonic["brier_score"],
            "ece": cal_curve_isotonic["expected_calibration_error"],
            "f1": metrics_isotonic["f1"],
            "precision": metrics_isotonic["precision"],
            "recall": metrics_isotonic["recall"],
            "roc_auc": metrics_isotonic.get("roc_auc"),
            "pr_auc": metrics_isotonic.get("pr_auc"),
            "calibration_curve": cal_curve_isotonic["calibration_curve"],
            "calibrator": cal_isotonic,
        },
    }

    # Identify best calibration method by Brier Score
    brier_scores = {
        "uncalibrated": cal_uncal["brier_score"],
        "sigmoid": cal_curve_sigmoid["brier_score"],
        "isotonic": cal_curve_isotonic["brier_score"],
    }
    best_method = min(brier_scores, key=brier_scores.get)
    results["best_method"] = best_method
    results["best_brier_score"] = brier_scores[best_method]

    logger.info(
        "Calibration Results: Uncalibrated Brier=%.4f, Sigmoid Brier=%.4f, Isotonic Brier=%.4f (Winner: %s)",
        brier_scores["uncalibrated"],
        brier_scores["sigmoid"],
        brier_scores["isotonic"],
        best_method.upper(),
    )

    return results
