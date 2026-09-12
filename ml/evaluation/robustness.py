"""
ETAFlow Robustness and Input Degradation Stress Testing Module.

Tests model resilience under synthetic perturbation:
1. Missing-data injection (5%, 10%, 20% MCAR missingness).
2. Environmental stress conditions (high traffic, severe weather shocks, poor road surfaces).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ml.preprocessing.pipeline import ETAPreprocessingPipeline
from ml.training.classification import predict_classification, predict_classification_proba
from ml.training.evaluation import evaluate_classification, evaluate_regression
from ml.training.regression import predict_regression

logger = logging.getLogger(__name__)


def inject_missingness(
    df: pd.DataFrame,
    missing_rate: float,
    random_state: int = 42,
    protected_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Inject Missing Completely At Random (MCAR) nulls into input features.

    Args:
        df: Input raw feature dataframe.
        missing_rate: Fraction of entries to mask as NaN (e.g. 0.05, 0.10, 0.20).
        random_state: Seed for reproducibility.
        protected_cols: Columns that must not be altered (e.g. ID, targets, dates).

    Returns:
        pd.DataFrame: Perturbed dataframe with added NaNs.
    """
    if missing_rate <= 0.0:
        return df.copy()

    df_corrupted = df.copy()
    rng = np.random.RandomState(random_state)

    protected = set(protected_cols or [
        "shipment_id",
        "order_date",
        "actual_dispatch_datetime",
        "actual_delivery_days",
        "is_delayed",
        "delivery_delay_days",
        "delivery_datetime",
    ])

    eligible_cols = [c for c in df_corrupted.columns if c not in protected]

    for col in eligible_cols:
        mask = rng.rand(len(df_corrupted)) < missing_rate
        df_corrupted.loc[mask, col] = np.nan

    return df_corrupted


def apply_environmental_shock(
    df: pd.DataFrame,
    shock_type: str,
) -> pd.DataFrame:
    """Apply systematic environmental stress scenario to dataframe.

    Supported shock_types:
    - "high_traffic": Sets congestion_index to 0.85 and traffic_level to "High".
    - "severe_weather": Sets weather_risk_score to 0.85 and weather_condition to "Severe".
    - "poor_road": Sets road_condition to "Poor".
    """
    df_shocked = df.copy()

    if shock_type == "high_traffic":
        if "congestion_index" in df_shocked.columns:
            df_shocked["congestion_index"] = 0.85
        if "traffic_level" in df_shocked.columns:
            df_shocked["traffic_level"] = "High"

    elif shock_type == "severe_weather":
        if "weather_risk_score" in df_shocked.columns:
            df_shocked["weather_risk_score"] = 0.85
        if "weather_condition" in df_shocked.columns:
            df_shocked["weather_condition"] = "Severe"

    elif shock_type == "poor_road":
        if "road_condition" in df_shocked.columns:
            df_shocked["road_condition"] = "Poor"

    else:
        raise ValueError(f"Unknown shock_type: {shock_type}")

    return df_shocked


def run_robustness_audit(
    model: Any,
    pipeline: ETAPreprocessingPipeline,
    df_clean: pd.DataFrame,
    target_type: str,
    missingness_rates: Optional[List[float]] = None,
    environmental_shocks: Optional[List[str]] = None,
    threshold: float = 0.50,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Execute complete robustness and stress testing battery.

    Args:
        model: Fitted estimator.
        pipeline: Fitted ETAPreprocessingPipeline.
        df_clean: Clean evaluation split (e.g. Validation dataframe).
        target_type: "regression" or "classification".
        missingness_rates: List of null injection fractions.
        environmental_shocks: List of environmental shock scenario names.
        threshold: Classification decision boundary.
        random_state: Random seed.

    Returns:
        Dict[str, Any]: Baseline metrics and delta changes under each stress test.
    """
    if missingness_rates is None:
        missingness_rates = [0.05, 0.10, 0.20]
    if environmental_shocks is None:
        environmental_shocks = ["high_traffic", "severe_weather", "poor_road"]

    target_col = "actual_delivery_days" if target_type == "regression" else "is_delayed"
    y_true = df_clean[target_col].values

    # 1. Baseline Performance
    X_clean = pipeline.transform(df_clean)
    if target_type == "regression":
        base_preds = predict_regression(model, X_clean)
        base_metrics = evaluate_regression(y_true, base_preds)
        base_primary = base_metrics["mae"]
        primary_name = "mae"
    else:
        base_prob = predict_classification_proba(model, X_clean)
        base_preds = (base_prob >= threshold).astype(int)
        base_metrics = evaluate_classification(y_true, base_preds, y_prob=base_prob)
        base_primary = base_metrics["f1"]
        primary_name = "f1"

    results: Dict[str, Any] = {
        "target_type": target_type,
        "primary_metric": primary_name,
        "baseline_score": base_primary,
        "baseline_metrics": base_metrics,
        "missingness_stress": [],
        "environmental_shocks": [],
    }

    # 2. Missing-Data Stress Tests
    for rate in missingness_rates:
        df_corrupt = inject_missingness(df_clean, rate, random_state=random_state)
        X_corrupt = pipeline.transform(df_corrupt)

        if target_type == "regression":
            preds = predict_regression(model, X_corrupt)
            m = evaluate_regression(y_true, preds)
            score = m["mae"]
        else:
            prob = predict_classification_proba(model, X_corrupt)
            preds = (prob >= threshold).astype(int)
            m = evaluate_classification(y_true, preds, y_prob=prob)
            score = m["f1"]

        abs_diff = score - base_primary
        rel_diff_pct = (abs_diff / base_primary) * 100 if base_primary != 0 else 0.0

        results["missingness_stress"].append({
            "missing_rate": rate,
            "missing_pct_label": f"{int(rate * 100)}%",
            "stressed_score": round(score, 4),
            "absolute_delta": round(abs_diff, 4),
            "relative_delta_pct": round(rel_diff_pct, 2),
            "metrics": m,
        })

    # 3. Environmental Shocks
    for shock in environmental_shocks:
        df_shocked = apply_environmental_shock(df_clean, shock)
        X_shocked = pipeline.transform(df_shocked)

        if target_type == "regression":
            preds = predict_regression(model, X_shocked)
            m = evaluate_regression(y_true, preds)
            score = m["mae"]
        else:
            prob = predict_classification_proba(model, X_shocked)
            preds = (prob >= threshold).astype(int)
            m = evaluate_classification(y_true, preds, y_prob=prob)
            score = m["f1"]

        abs_diff = score - base_primary
        rel_diff_pct = (abs_diff / base_primary) * 100 if base_primary != 0 else 0.0

        results["environmental_shocks"].append({
            "shock_type": shock,
            "stressed_score": round(score, 4),
            "absolute_delta": round(abs_diff, 4),
            "relative_delta_pct": round(rel_diff_pct, 2),
            "metrics": m,
        })

    return results
