"""
ETAFlow Baseline Classification Suite.

Implements model family definitions and training runners for delay risk classification:
- Level 0: DummyClassifier
- Level 1: Logistic Regression (StandardScaler conditioned)
- Level 2: RandomForestClassifier
- Level 3: LightGBM Classifier
- Level 3: XGBoost Classifier
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from ml.training.model_utils import measure_execution_time

logger = logging.getLogger(__name__)


def get_classification_models(
    config: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Instantiate the standardized M6-A baseline classification model suite.

    Args:
        config: Optional configuration dictionary containing model hyperparameters.
        random_state: Random seed for stochastic models.

    Returns:
        Dict[str, Any]: Dictionary mapping model keys to model instances/pipelines.
    """
    cfg = config or {}
    models_cfg = cfg.get("models", {}).get("classification", {})

    dummy_cfg = models_cfg.get("dummy", {})
    lr_cfg = models_cfg.get("logistic_regression", {})
    rf_cfg = models_cfg.get("random_forest", {})
    lgbm_cfg = models_cfg.get("lightgbm", {})
    xgb_cfg = models_cfg.get("xgboost", {})

    models: Dict[str, Any] = {}

    # Level 0: Dummy
    models["dummy"] = DummyClassifier(
        strategy=dummy_cfg.get("strategy", "prior"),
        random_state=random_state,
    )

    # Level 1: Logistic Regression (wrapped with StandardScaler for numerical convergence)
    max_iter = int(lr_cfg.get("max_iter", 500))
    models["logistic_regression"] = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=False)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=max_iter,
                    random_state=random_state,
                ),
            ),
        ]
    )

    # Level 2: Random Forest
    models["random_forest"] = RandomForestClassifier(
        n_estimators=int(rf_cfg.get("n_estimators", 100)),
        max_depth=rf_cfg.get("max_depth", 12),
        random_state=int(rf_cfg.get("random_state", random_state)),
        n_jobs=int(rf_cfg.get("n_jobs", -1)),
    )

    # Level 3: LightGBM
    models["lightgbm"] = LGBMClassifier(
        n_estimators=int(lgbm_cfg.get("n_estimators", 100)),
        learning_rate=float(lgbm_cfg.get("learning_rate", 0.1)),
        random_state=int(lgbm_cfg.get("random_state", random_state)),
        n_jobs=int(lgbm_cfg.get("n_jobs", -1)),
        verbose=int(lgbm_cfg.get("verbose", -1)),
    )

    # Level 3: XGBoost
    models["xgboost"] = XGBClassifier(
        n_estimators=int(xgb_cfg.get("n_estimators", 100)),
        max_depth=int(xgb_cfg.get("max_depth", 6)),
        learning_rate=float(xgb_cfg.get("learning_rate", 0.1)),
        random_state=int(xgb_cfg.get("random_state", random_state)),
        n_jobs=int(xgb_cfg.get("n_jobs", -1)),
        eval_metric=xgb_cfg.get("eval_metric", "logloss"),
    )

    return models


def train_classification_model(
    model_name: str,
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Tuple[Any, float]:
    """Fit a classification model and measure training duration.

    Args:
        model_name: Display identifier for the model.
        model: Estimator or Pipeline to train.
        X_train: Training feature matrix.
        y_train: Training binary target values (is_delayed).

    Returns:
        Tuple[Any, float]: (fitted_model, duration_seconds).
    """
    logger.info("Training classification model '%s' on %d samples...", model_name, len(X_train))
    with measure_execution_time() as timer:
        model.fit(X_train, y_train)

    duration = timer["duration_seconds"]
    logger.info("Fitted '%s' in %.2f seconds.", model_name, duration)
    return model, duration


def predict_classification(
    model: Any,
    X: pd.DataFrame,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Generate discrete delay classification predictions and continuous probabilities.

    Args:
        model: Fitted estimator or Pipeline.
        X: Input feature matrix.

    Returns:
        Tuple[np.ndarray, Optional[np.ndarray]]: (discrete_predictions, positive_class_probabilities).
    """
    preds = model.predict(X)
    y_pred = np.asarray(preds, dtype=np.int64)

    y_prob: Optional[np.ndarray] = None
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(X)
            # If binary classes [0, 1], positive class is column index 1
            if probs.ndim == 2 and probs.shape[1] >= 2:
                y_prob = np.asarray(probs[:, 1], dtype=np.float64)
            elif probs.ndim == 2 and probs.shape[1] == 1:
                y_prob = np.asarray(probs[:, 0], dtype=np.float64)
        except Exception as e:
            logger.warning("predict_proba failed for model: %s", e)

    return y_pred, y_prob


def predict_classification_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Generate positive-class delay probabilities.

    Args:
        model: Fitted estimator or Pipeline.
        X: Input feature matrix.

    Returns:
        np.ndarray: 1D array of positive class probabilities.
    """
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        if probs.ndim == 2 and probs.shape[1] >= 2:
            return np.asarray(probs[:, 1], dtype=np.float64)
        elif probs.ndim == 2 and probs.shape[1] == 1:
            return np.asarray(probs[:, 0], dtype=np.float64)
        return np.asarray(probs, dtype=np.float64)
    # Fallback to discrete predictions if model lacks predict_proba
    return np.asarray(model.predict(X), dtype=np.float64)
