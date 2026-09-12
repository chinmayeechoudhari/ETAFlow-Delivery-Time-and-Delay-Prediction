"""
ETAFlow Model Utilities Module.

Provides serialization, model loading, feature importance extraction,
timing measurements, and deterministic model selection logic for M6-A benchmarking.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


@contextmanager
def measure_execution_time() -> Generator[Dict[str, float], None, None]:
    """Context manager to measure runtime duration in seconds."""
    timing_info: Dict[str, float] = {"duration_seconds": 0.0}
    start = time.perf_counter()
    try:
        yield timing_info
    finally:
        timing_info["duration_seconds"] = float(time.perf_counter() - start)


def save_model(model: Any, filepath: Union[str, Path]) -> Path:
    """Serialize a trained model or pipeline to disk using joblib.

    Args:
        model: Trained estimator or pipeline.
        filepath: Target destination path.

    Returns:
        Path: Resolved output path.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Model successfully saved to %s", path)
    return path


def load_model(filepath: Union[str, Path]) -> Any:
    """Load a serialized model or pipeline from disk using joblib.

    Args:
        filepath: Path to serialized artifact.

    Returns:
        Any: Deserialized model object.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at: {path}")
    model = joblib.load(path)
    logger.info("Model loaded successfully from %s", path)
    return model


def extract_feature_importance(
    model: Any,
    feature_names: List[str],
    top_k: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    """Extract and rank feature importances from tree-based or linear models.

    Args:
        model: Estimator or Pipeline.
        feature_names: Ordered list of feature column names.
        top_k: Optional limit on number of top features to return.

    Returns:
        Optional[pd.DataFrame]: DataFrame with columns ['feature', 'importance'],
        or None if model does not expose feature importances or coefficients.
    """
    # If pipeline, extract the terminal estimator
    estimator = model
    if isinstance(model, Pipeline):
        estimator = model.steps[-1][1]

    importances: Optional[np.ndarray] = None

    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        coef = estimator.coef_
        if coef.ndim > 1:
            importances = np.abs(coef[0])
        else:
            importances = np.abs(coef)

    if importances is None or len(importances) != len(feature_names):
        return None

    df_imp = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    )
    df_imp = df_imp.sort_values(by="importance", ascending=False).reset_index(drop=True)

    if top_k is not None and top_k > 0:
        df_imp = df_imp.head(top_k)

    return df_imp


def select_best_regression_model(
    results: List[Dict[str, Any]],
    metric_key: str = "mae",
) -> Dict[str, Any]:
    """Deterministically select best baseline regression candidate.

    Primary metric: MAE (lower is better)
    Tie-breakers: RMSE (lower is better), R2 (higher is better), Within_1_Day (higher is better)

    Args:
        results: List of benchmark result dictionaries for regression.
        metric_key: Primary sorting metric.

    Returns:
        Dict[str, Any]: The winning result dictionary.
    """
    if not results:
        raise ValueError("Cannot select best model from empty results list.")

    def sort_key(r: Dict[str, Any]) -> Tuple[float, float, float, float]:
        metrics = r.get("test_metrics", r.get("val_metrics", r))
        mae = metrics.get("mae", float("inf"))
        rmse = metrics.get("rmse", float("inf"))
        r2 = -metrics.get("r2", float("-inf"))
        within_1 = -metrics.get("within_1_day", float("-inf"))
        return (mae, rmse, r2, within_1)

    sorted_results = sorted(results, key=sort_key)
    return sorted_results[0]


def select_best_classification_model(
    results: List[Dict[str, Any]],
    metric_key: str = "f1",
) -> Dict[str, Any]:
    """Deterministically select best baseline classification candidate.

    Primary metric: F1 (higher is better)
    Tie-breakers: PR-AUC (higher is better), ROC-AUC (higher is better), Recall (higher is better)

    Args:
        results: List of benchmark result dictionaries for classification.
        metric_key: Primary sorting metric.

    Returns:
        Dict[str, Any]: The winning result dictionary.
    """
    if not results:
        raise ValueError("Cannot select best model from empty results list.")

    def sort_key(r: Dict[str, Any]) -> Tuple[float, float, float, float]:
        metrics = r.get("test_metrics", r.get("val_metrics", r))
        f1 = -metrics.get("f1", float("-inf"))
        pr_auc = -metrics.get("pr_auc", float("-inf")) if metrics.get("pr_auc") is not None else 0.0
        roc_auc = -metrics.get("roc_auc", float("-inf")) if metrics.get("roc_auc") is not None else 0.0
        recall = -metrics.get("recall", float("-inf"))
        return (f1, pr_auc, roc_auc, recall)

    sorted_results = sorted(results, key=sort_key)
    return sorted_results[0]
