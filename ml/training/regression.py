"""
ETAFlow Baseline Regression Suite.

Implements model family definitions and training runners for delivery-time regression:
- Level 0: DummyRegressor
- Level 1: Ridge Regression (StandardScaler conditioned)
- Level 2: RandomForestRegressor
- Level 3: LightGBM Regressor
- Level 3: XGBoost Regressor
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from ml.training.model_utils import measure_execution_time

logger = logging.getLogger(__name__)


def get_regression_models(
    config: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Instantiate the standardized M6-A baseline regression model suite.

    Args:
        config: Optional configuration dictionary containing model hyperparameters.
        random_state: Random seed for stochastic models.

    Returns:
        Dict[str, Any]: Dictionary mapping model keys to model instances/pipelines.
    """
    cfg = config or {}
    models_cfg = cfg.get("models", {}).get("regression", {})

    dummy_cfg = models_cfg.get("dummy", {})
    ridge_cfg = models_cfg.get("ridge", {})
    rf_cfg = models_cfg.get("random_forest", {})
    lgbm_cfg = models_cfg.get("lightgbm", {})
    xgb_cfg = models_cfg.get("xgboost", {})

    models: Dict[str, Any] = {}

    # Level 0: Dummy
    models["dummy"] = DummyRegressor(
        strategy=dummy_cfg.get("strategy", "mean")
    )

    # Level 1: Ridge (wrapped with StandardScaler for numerical stability across 200+ features)
    ridge_alpha = float(ridge_cfg.get("alpha", 1.0))
    models["ridge"] = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=False)),
            ("regressor", Ridge(alpha=ridge_alpha, random_state=random_state)),
        ]
    )

    # Level 2: Random Forest
    models["random_forest"] = RandomForestRegressor(
        n_estimators=int(rf_cfg.get("n_estimators", 100)),
        max_depth=rf_cfg.get("max_depth", 12),
        random_state=int(rf_cfg.get("random_state", random_state)),
        n_jobs=int(rf_cfg.get("n_jobs", -1)),
    )

    # Level 3: LightGBM
    models["lightgbm"] = LGBMRegressor(
        n_estimators=int(lgbm_cfg.get("n_estimators", 100)),
        learning_rate=float(lgbm_cfg.get("learning_rate", 0.1)),
        random_state=int(lgbm_cfg.get("random_state", random_state)),
        n_jobs=int(lgbm_cfg.get("n_jobs", -1)),
        verbose=int(lgbm_cfg.get("verbose", -1)),
    )

    # Level 3: XGBoost
    models["xgboost"] = XGBRegressor(
        n_estimators=int(xgb_cfg.get("n_estimators", 100)),
        max_depth=int(xgb_cfg.get("max_depth", 6)),
        learning_rate=float(xgb_cfg.get("learning_rate", 0.1)),
        random_state=int(xgb_cfg.get("random_state", random_state)),
        n_jobs=int(xgb_cfg.get("n_jobs", -1)),
    )

    return models


def train_regression_model(
    model_name: str,
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Tuple[Any, float]:
    """Fit a regression model and measure training duration.

    Args:
        model_name: Display identifier for the model.
        model: Estimator or Pipeline to train.
        X_train: Training feature matrix.
        y_train: Training target values (actual_delivery_days).

    Returns:
        Tuple[Any, float]: (fitted_model, duration_seconds).
    """
    logger.info("Training regression model '%s' on %d samples...", model_name, len(X_train))
    with measure_execution_time() as timer:
        model.fit(X_train, y_train)

    duration = timer["duration_seconds"]
    logger.info("Fitted '%s' in %.2f seconds.", model_name, duration)
    return model, duration


def predict_regression(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Generate delivery-time regression predictions.

    Args:
        model: Fitted estimator or Pipeline.
        X: Input feature matrix.

    Returns:
        np.ndarray: 1D array of predicted delivery days.
    """
    preds = model.predict(X)
    return np.asarray(preds, dtype=np.float64)
