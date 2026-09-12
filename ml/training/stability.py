"""
ETAFlow Multi-Seed Model Stability Audit Module.

Evaluates candidate model performance variance across 5 controlled random seeds:
[42, 123, 2024, 3407, 999] to ensure training stability and eliminate seed reliance.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ml.training.classification import (
    LGBMClassifier,
    XGBClassifier,
    predict_classification,
    predict_classification_proba,
)
from ml.training.evaluation import evaluate_classification, evaluate_regression
from ml.training.regression import LGBMRegressor, XGBRegressor, predict_regression
from ml.training.time_cv import instantiate_model

logger = logging.getLogger(__name__)


def evaluate_model_stability(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    model_name: str,
    target_type: str,
    model_params: Optional[Dict[str, Any]] = None,
    seeds: Optional[List[int]] = None,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """Train candidate model across multiple random seeds and compute variance metrics.

    Args:
        X_train: Preprocessed train features.
        y_train: Train target array.
        X_val: Preprocessed validation features.
        y_val: Validation target array.
        model_name: "lightgbm" or "xgboost".
        target_type: "regression" or "classification".
        model_params: Hyperparameters for the model.
        seeds: List of seeds to evaluate (default: [42, 123, 2024, 3407, 999]).
        threshold: Classification decision boundary.

    Returns:
        Dict[str, Any]: Per-seed metric records and aggregate mean, std, min, max stats.
    """
    if seeds is None:
        seeds = [42, 123, 2024, 3407, 999]

    seed_runs: List[Dict[str, Any]] = []

    logger.info(
        "Running stability audit across %d seeds for %s | %s...",
        len(seeds),
        model_name.upper(),
        target_type.upper(),
    )

    for seed in seeds:
        model = instantiate_model(
            model_name=model_name,
            target_type=target_type,
            params=model_params,
            random_state=seed,
        )

        model.fit(X_train, y_train)

        if target_type == "regression":
            preds = predict_regression(model, X_val)
            metrics = evaluate_regression(y_val, preds)
            seed_runs.append({
                "seed": seed,
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
                "within_1_day": metrics["within_1_day"],
            })
        else:
            prob = predict_classification_proba(model, X_val)
            preds = (prob >= threshold).astype(int)
            metrics = evaluate_classification(y_val, preds, y_prob=prob)
            seed_runs.append({
                "seed": seed,
                "f1": metrics["f1"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "roc_auc": metrics.get("roc_auc", 0.0),
                "pr_auc": metrics.get("pr_auc", 0.0),
            })

    # Aggregate statistics
    metric_keys = [k for k in seed_runs[0].keys() if k != "seed"]
    aggregate: Dict[str, Dict[str, float]] = {}

    for k in metric_keys:
        vals = [run[k] for run in seed_runs]
        aggregate[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "range": float(np.max(vals) - np.min(vals)),
        }

    # Assess stability
    primary_metric = "mae" if target_type == "regression" else "f1"
    primary_std = aggregate[primary_metric]["std"]
    is_stable = bool(primary_std < 0.02)  # Low standard deviation threshold

    return {
        "model_name": model_name,
        "target_type": target_type,
        "seeds_evaluated": seeds,
        "primary_metric": primary_metric,
        "is_stable": is_stable,
        "seed_runs": seed_runs,
        "aggregate": aggregate,
    }
