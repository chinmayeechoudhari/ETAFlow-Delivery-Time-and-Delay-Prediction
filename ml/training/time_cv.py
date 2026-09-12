"""
ETAFlow Time-Aware Cross Validation Module.

Implements expanding-window chronological cross-validation to evaluate models
without future-to-past data leakage. Guarantees that preprocessing transformers
are fitted strictly within each fold's training slice.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor

from ml.preprocessing.pipeline import ETAPreprocessingPipeline, build_preprocessing_pipeline
from ml.training.classification import (
    predict_classification,
    predict_classification_proba,
)
from ml.training.evaluation import evaluate_classification, evaluate_regression
from ml.training.regression import predict_regression

logger = logging.getLogger(__name__)


class TimeAwareExpandingWindowCV:
    """Chronological expanding-window cross-validation generator.

    Splits chronologically sorted data such that training data strictly precedes
    validation data, expanding the training window across successive folds.
    """

    def __init__(
        self,
        n_splits: int = 3,
        min_train_size: Optional[int] = None,
        min_train_ratio: float = 0.5,
    ) -> None:
        """Initialize expanding-window chronological CV.

        Args:
            n_splits: Number of chronological folds (must be >= 2).
            min_train_size: Minimum number of samples in the initial training fold.
            min_train_ratio: If min_train_size is None, initial fold fraction of dataset.
        """
        if n_splits < 2:
            raise ValueError(f"n_splits must be at least 2, got {n_splits}")
        self.n_splits = n_splits
        self.min_train_size = min_train_size
        self.min_train_ratio = min_train_ratio

    def split(
        self, df: pd.DataFrame
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generate (train_idx, val_idx) indices for each chronological fold.

        Args:
            df: DataFrame sorted chronologically.

        Yields:
            Tuple[np.ndarray, np.ndarray]: Integer position indices for (train, val).
        """
        n_samples = len(df)
        initial_train = (
            self.min_train_size
            if self.min_train_size is not None
            else int(n_samples * self.min_train_ratio)
        )

        if initial_train <= 0 or initial_train >= n_samples:
            raise ValueError(
                f"Invalid initial_train_size {initial_train} for {n_samples} samples."
            )

        remaining_samples = n_samples - initial_train
        val_step = remaining_samples // self.n_splits

        if val_step <= 0:
            raise ValueError(
                f"Remaining samples ({remaining_samples}) too small for {self.n_splits} splits."
            )

        for fold_idx in range(self.n_splits):
            train_end = initial_train + fold_idx * val_step
            val_end = (
                train_end + val_step
                if fold_idx < self.n_splits - 1
                else n_samples
            )

            train_indices = np.arange(0, train_end)
            val_indices = np.arange(train_end, val_end)

            # Strict temporal invariant assertion
            assert len(train_indices) > 0, "Train partition cannot be empty."
            assert len(val_indices) > 0, "Val partition cannot be empty."
            assert train_indices.max() < val_indices.min(), (
                f"Temporal leakage detected: max train index {train_indices.max()} "
                f">= min val index {val_indices.min()}."
            )

            yield train_indices, val_indices


def instantiate_model(
    model_name: str,
    target_type: str,
    params: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
) -> Any:
    """Instantiate a supported tree model with specified parameters.

    Args:
        model_name: "lightgbm" or "xgboost".
        target_type: "regression" or "classification".
        params: Optional hyperparameter overrides.
        random_state: Seed for reproducibility.

    Returns:
        Estimator instance.
    """
    m_name = model_name.lower()
    t_type = target_type.lower()
    p = dict(params or {})

    if t_type == "regression":
        if m_name == "lightgbm":
            base_p = {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "random_state": random_state,
                "n_jobs": -1,
                "verbose": -1,
            }
            base_p.update(p)
            return LGBMRegressor(**base_p)
        elif m_name == "xgboost":
            base_p = {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 6,
                "random_state": random_state,
                "n_jobs": -1,
            }
            base_p.update(p)
            return XGBRegressor(**base_p)
        else:
            raise ValueError(f"Unsupported regression model: {model_name}")

    elif t_type == "classification":
        if m_name == "lightgbm":
            base_p = {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "random_state": random_state,
                "n_jobs": -1,
                "verbose": -1,
            }
            base_p.update(p)
            return LGBMClassifier(**base_p)
        elif m_name == "xgboost":
            base_p = {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 6,
                "random_state": random_state,
                "n_jobs": -1,
            }
            base_p.update(p)
            return XGBClassifier(**base_p)
        else:
            raise ValueError(f"Unsupported classification model: {model_name}")
    else:
        raise ValueError(f"Unknown target_type: {target_type}")


def run_time_aware_cv(
    df_train: pd.DataFrame,
    horizon: str,
    target_type: str,
    model_name: str,
    model_params: Optional[Dict[str, Any]] = None,
    n_splits: int = 3,
    min_train_size: Optional[int] = 35000,
    random_state: int = 42,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Execute expanding-window chronological cross-validation.

    Args:
        df_train: Chronologically sorted training split (e.g., 70,000 records).
        horizon: Operational horizon ("booking" or "dispatch").
        target_type: "regression" or "classification".
        model_name: "lightgbm" or "xgboost".
        model_params: Hyperparameter dictionary for model instantiation.
        n_splits: Number of expanding folds.
        min_train_size: Minimum initial fold training size.
        random_state: Seed for reproducibility.
        threshold: Classification probability threshold.

    Returns:
        Dict[str, Any]: Fold-level and aggregated mean/std cross-validation metrics.
    """
    target_col = (
        "actual_delivery_days" if target_type == "regression" else "is_delayed"
    )

    cv = TimeAwareExpandingWindowCV(
        n_splits=n_splits,
        min_train_size=min_train_size,
    )

    fold_metrics: List[Dict[str, Any]] = []

    logger.info(
        "Starting %d-fold Time-Aware CV for %s | %s | %s...",
        n_splits,
        horizon.upper(),
        target_type.upper(),
        model_name.upper(),
    )

    for fold_num, (train_idx, val_idx) in enumerate(cv.split(df_train), start=1):
        df_f_train = df_train.iloc[train_idx].copy()
        df_f_val = df_train.iloc[val_idx].copy()

        # Fit preprocessing strictly within this fold's training slice
        cfg_path = "configs/features_booking.yaml" if horizon.lower() == "booking" else "configs/features_dispatch.yaml"
        pipeline = build_preprocessing_pipeline(cfg_path)
        X_f_train = pipeline.fit_transform(df_f_train)
        X_f_val = pipeline.transform(df_f_val)

        y_f_train = df_f_train[target_col].values
        y_f_val = df_f_val[target_col].values

        model = instantiate_model(
            model_name=model_name,
            target_type=target_type,
            params=model_params,
            random_state=random_state,
        )

        model.fit(X_f_train, y_f_train)

        if target_type == "regression":
            preds = predict_regression(model, X_f_val)
            metrics = evaluate_regression(y_f_val, preds)
        else:
            proba = predict_classification_proba(model, X_f_val)
            preds = (proba >= threshold).astype(int)
            metrics = evaluate_classification(y_f_val, preds, y_prob=proba)

        fold_record = {
            "fold": fold_num,
            "train_samples": len(train_idx),
            "val_samples": len(val_idx),
            "metrics": metrics,
        }
        fold_metrics.append(fold_record)

        if target_type == "regression":
            logger.info(
                "Fold %d: Train=%d, Val=%d -> MAE=%.4f, RMSE=%.4f, R2=%.4f",
                fold_num,
                len(train_idx),
                len(val_idx),
                metrics["mae"],
                metrics["rmse"],
                metrics["r2"],
            )
        else:
            logger.info(
                "Fold %d: Train=%d, Val=%d -> F1=%.4f, ROC-AUC=%.4f, PR-AUC=%.4f",
                fold_num,
                len(train_idx),
                len(val_idx),
                metrics["f1"],
                metrics.get("roc_auc", 0.0),
                metrics.get("pr_auc", 0.0),
            )

    # Compute aggregate summary statistics (mean and std)
    summary: Dict[str, Dict[str, float]] = {}
    metric_keys = list(fold_metrics[0]["metrics"].keys())

    for k in metric_keys:
        # Filter numeric scalar values only
        vals = [
            f["metrics"][k]
            for f in fold_metrics
            if isinstance(f["metrics"].get(k), (int, float, np.number))
        ]
        if len(vals) == len(fold_metrics):
            summary[k] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }

    return {
        "horizon": horizon,
        "target_type": target_type,
        "model_name": model_name,
        "n_splits": n_splits,
        "fold_results": fold_metrics,
        "aggregate": summary,
    }
