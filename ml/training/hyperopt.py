"""
ETAFlow Hyperparameter Optimization Module.

Implements Bayesian optimization via Optuna for LightGBM and XGBoost.
Evaluates candidates strictly on chronological validation partitions
with zero exposure to the final quarantined test set.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

import numpy as np
import optuna  # pyrefly: ignore [missing-import]
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor

from ml.training.classification import predict_classification
from ml.training.evaluation import evaluate_classification, evaluate_regression
from ml.training.regression import predict_regression

# Suppress Optuna verbose logging by default
optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)


def sample_lgbm_params(
    trial: optuna.Trial,
    target_type: str,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Sample LightGBM hyperparameters within bounded search space."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 350, step=50),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "random_state": random_state,
        "n_jobs": -1,
        "verbose": -1,
    }


def sample_xgb_params(
    trial: optuna.Trial,
    target_type: str,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Sample XGBoost hyperparameters within bounded search space."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 350, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "random_state": random_state,
        "n_jobs": -1,
    }


def optimize_hyperparameters(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    model_name: str,
    target_type: str,
    n_trials: int = 20,
    timeout_seconds: Optional[int] = 600,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Run Optuna Bayesian hyperparameter optimization.

    Args:
        X_train: Preprocessed training feature matrix.
        y_train: Training target array.
        X_val: Preprocessed validation feature matrix.
        y_val: Validation target array.
        model_name: "lightgbm" or "xgboost".
        target_type: "regression" or "classification".
        n_trials: Number of optimization trials.
        timeout_seconds: Maximum duration in seconds.
        random_state: Seed for sampler.

    Returns:
        Dict[str, Any]: Best parameters, best score, trial history, and metadata.
    """
    m_name = model_name.lower()
    t_type = target_type.lower()
    sampler = optuna.samplers.TPESampler(seed=random_state)

    is_regression = t_type == "regression"
    direction = "minimize" if is_regression else "maximize"
    study_name = f"{m_name}_{t_type}_opt"

    study = optuna.create_study(
        direction=direction,
        sampler=sampler,
        study_name=study_name,
    )

    def objective(trial: optuna.Trial) -> float:
        if m_name == "lightgbm":
            params = sample_lgbm_params(trial, t_type, random_state=random_state)
            model = (
                LGBMRegressor(**params) if is_regression else LGBMClassifier(**params)
            )
        elif m_name == "xgboost":
            params = sample_xgb_params(trial, t_type, random_state=random_state)
            model = (
                XGBRegressor(**params) if is_regression else XGBClassifier(**params)
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        model.fit(X_train, y_train)

        if is_regression:
            preds = predict_regression(model, X_val)
            metrics = evaluate_regression(y_val, preds)
            return float(metrics["mae"])  # Minimize MAE
        else:
            preds, prob = predict_classification(model, X_val)
            metrics = evaluate_classification(y_val, preds, y_prob=prob)
            return float(metrics["f1"])  # Maximize F1

    logger.info(
        "Starting %s (%d trials) for %s | %s...",
        study_name,
        n_trials,
        model_name.upper(),
        target_type.upper(),
    )

    start_time = time.time()
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout_seconds,
        show_progress_bar=False,
    )
    duration = time.time() - start_time

    best_trial = study.best_trial
    logger.info(
        "Optuna completed in %.1fs. Best trial #%d: Value=%.4f",
        duration,
        best_trial.number,
        best_trial.value,
    )

    # Clean up column names for readability
    trials_records = [
        {
            "trial_number": t.number,
            "value": t.value,
            "params": t.params,
            "state": str(t.state),
        }
        for t in study.trials
    ]

    return {
        "model_name": model_name,
        "target_type": target_type,
        "optimization_direction": direction,
        "objective_metric": "mae" if is_regression else "f1",
        "best_trial_number": best_trial.number,
        "best_value": float(best_trial.value),
        "best_params": best_trial.params,
        "n_trials_completed": len(study.trials),
        "duration_seconds": duration,
        "trials_records": trials_records,
    }
