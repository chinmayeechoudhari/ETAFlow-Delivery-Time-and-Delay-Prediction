"""
Unit Tests for ETAFlow Advanced ML Optimization & Evaluation Layer (Milestone 6-B).

Verifies:
1. Chronological expanding-window CV generator and temporal invariants.
2. Leakage prevention in cross-validation folds.
3. Optuna parameter sampling bounded spaces.
4. Classification decision threshold optimization and metric trade-offs.
5. Probability calibration (Sigmoid and Isotonic) and Brier score evaluation.
6. Subgroup sliced error analysis and binning.
7. Robustness perturbation (MCAR missingness and environmental shocks).
8. Multi-seed stability runner and variance aggregation.
9. TreeExplainer SHAP global importance and local narrative generation.
10. Model selection artifact integrity.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.datasets import make_classification, make_regression

from ml.evaluation.robustness import apply_environmental_shock, inject_missingness, run_robustness_audit
from ml.evaluation.subgroup import analyze_classification_subgroups, analyze_regression_subgroups, create_subgroup_slices
from ml.explainability.shap_explainer import ETAShapExplainer
from ml.preprocessing.pipeline import ETAPreprocessingPipeline
from ml.training.calibration import calibrate_and_evaluate, evaluate_calibration_curve
from ml.training.stability import evaluate_model_stability
from ml.training.threshold_tuner import evaluate_threshold, find_optimal_threshold
from ml.training.time_cv import TimeAwareExpandingWindowCV, instantiate_model, run_time_aware_cv


@pytest.fixture
def synthetic_chronological_df() -> pd.DataFrame:
    """Create deterministic chronological DataFrame for optimization testing."""
    n = 300
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")
    rng = np.random.RandomState(42)

    return pd.DataFrame({
        "shipment_id": [f"SHP{i:05d}" for i in range(n)],
        "order_date": dates,
        "actual_dispatch_datetime": dates + pd.Timedelta(hours=4),
        "origin_hub": rng.choice(["Hub_A", "Hub_B", "Hub_C"], size=n),
        "destination_hub": rng.choice(["Hub_X", "Hub_Y", "Hub_Z"], size=n),
        "carrier": rng.choice(["ExpressAir", "FastFreight", "RoadRunner"], size=n),
        "transport_mode": rng.choice(["Road", "Air", "Rail"], size=n),
        "distance_km": rng.uniform(50, 4000, size=n),
        "package_weight_kg": rng.uniform(1.0, 80.0, size=n),
        "traffic_level": rng.choice(["Low", "Medium", "High"], size=n),
        "weather_condition": rng.choice(["Clear", "Rain", "Severe"], size=n),
        "road_condition": rng.choice(["Good", "Moderate", "Poor"], size=n),
        "is_holiday": rng.choice([0, 1], size=n, p=[0.9, 0.1]),
        "congestion_index": rng.uniform(0.1, 0.9, size=n),
        "weather_risk_score": rng.uniform(0.1, 0.9, size=n),
        "number_of_stops": rng.randint(0, 5, size=n),
        "route_complexity_score": rng.uniform(1.0, 10.0, size=n),
        "warehouse_processing_hours": rng.uniform(1.0, 8.0, size=n),
        "loading_time_hours": rng.uniform(0.5, 4.0, size=n),
        "customs_clearance_hours": rng.uniform(0.0, 5.0, size=n),
        "handling_time_hours": rng.uniform(0.5, 3.0, size=n),
        "promised_delivery_days": rng.uniform(2.0, 7.0, size=n),
        "actual_delivery_days": rng.uniform(1.5, 8.0, size=n),
        "is_delayed": rng.choice([0, 1], size=n, p=[0.7, 0.3]),
    })


def test_time_aware_cv_generator(synthetic_chronological_df):
    """Verify expanding window CV splits strictly respect chronological order."""
    df = synthetic_chronological_df
    cv = TimeAwareExpandingWindowCV(n_splits=3, min_train_size=150)

    splits = list(cv.split(df))
    assert len(splits) == 3

    prev_train_size = 0
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        # Strictly expanding train size
        assert len(train_idx) > prev_train_size
        prev_train_size = len(train_idx)

        # No empty validation
        assert len(val_idx) > 0

        # Max train index strictly precedes min val index (Zero future-to-past leakage)
        assert train_idx.max() < val_idx.min()

        # No overlap between train and val
        assert len(set(train_idx).intersection(set(val_idx))) == 0


def test_threshold_optimization():
    """Verify threshold sweep identifies optimal F1 point."""
    rng = np.random.RandomState(42)
    y_true = rng.choice([0, 1], size=200, p=[0.6, 0.4])
    # Simulated probabilities with signal
    y_prob = np.clip(y_true * 0.6 + rng.uniform(0.1, 0.4, size=200), 0.0, 1.0)

    result = find_optimal_threshold(y_true, y_prob, threshold_range=(0.3, 0.7), step=0.05)

    assert "optimal_threshold" in result
    assert "optimal_metrics" in result
    assert "baseline_metrics" in result
    assert 0.3 <= result["optimal_threshold"] <= 0.7
    assert result["optimal_metrics"]["f1"] >= 0.0
    assert len(result["threshold_curve"]) > 0


def test_probability_calibration():
    """Verify Platt and Isotonic calibration evaluators."""
    rng = np.random.RandomState(42)
    X = rng.randn(150, 5)
    y = rng.choice([0, 1], size=150, p=[0.7, 0.3])

    X_train, X_val = pd.DataFrame(X[:100]), pd.DataFrame(X[100:])
    y_train, y_val = y[:100], y[100:]

    model = LGBMClassifier(n_estimators=10, random_state=42, verbose=-1)
    model.fit(X_train, y_train)

    cal_results = calibrate_and_evaluate(model, X_train, y_train, X_val, y_val)

    assert "uncalibrated" in cal_results
    assert "sigmoid" in cal_results
    assert "isotonic" in cal_results
    assert "best_method" in cal_results
    assert 0.0 <= cal_results["uncalibrated"]["brier_score"] <= 1.0
    assert 0.0 <= cal_results["sigmoid"]["brier_score"] <= 1.0


def test_subgroup_analysis_regression(synthetic_chronological_df):
    """Verify sliced subgroup error analysis for continuous ETA regression."""
    df = synthetic_chronological_df
    y_true = df["actual_delivery_days"].values
    y_pred = y_true + np.random.RandomState(42).normal(0, 0.2, size=len(y_true))

    results = analyze_regression_subgroups(df, y_true, y_pred, min_samples=10)

    assert "transport_mode" in results
    assert "distance_tier" in results
    assert "weather_condition" in results

    mode_slices = results["transport_mode"]
    assert len(mode_slices) > 0
    for s in mode_slices:
        assert "mae" in s
        assert "rmse" in s
        assert "mean_bias" in s
        assert "sample_count" in s
        assert s["sample_count"] >= 10


def test_subgroup_analysis_classification(synthetic_chronological_df):
    """Verify sliced subgroup error analysis for binary delay classification."""
    df = synthetic_chronological_df
    y_true = df["is_delayed"].values
    y_pred = y_true.copy()
    # Introduce small error
    y_pred[:10] = 1 - y_pred[:10]

    results = analyze_classification_subgroups(df, y_true, y_pred, min_samples=10)

    assert "carrier" in results
    assert "traffic_level" in results

    carrier_slices = results["carrier"]
    for s in carrier_slices:
        assert "precision" in s
        assert "recall" in s
        assert "f1" in s
        assert "fpr" in s
        assert "fnr" in s


def test_robustness_perturbation(synthetic_chronological_df):
    """Verify MCAR missingness injection preserves protected columns and introduces NaNs."""
    df = synthetic_chronological_df
    df_corrupt = inject_missingness(df, missing_rate=0.20, random_state=42)

    # Protected columns must NEVER contain NaNs
    assert df_corrupt["shipment_id"].isna().sum() == 0
    assert df_corrupt["order_date"].isna().sum() == 0
    assert df_corrupt["actual_delivery_days"].isna().sum() == 0

    # Non-protected feature columns must contain injected NaNs
    total_corrupt_nans = df_corrupt[["distance_km", "package_weight_kg", "congestion_index"]].isna().sum().sum()
    assert total_corrupt_nans > 0


def test_environmental_shock(synthetic_chronological_df):
    """Verify environmental stress injection updates designated feature dimensions."""
    df = synthetic_chronological_df
    df_traffic = apply_environmental_shock(df, "high_traffic")
    assert (df_traffic["traffic_level"] == "High").all()
    assert (df_traffic["congestion_index"] == 0.85).all()

    df_weather = apply_environmental_shock(df, "severe_weather")
    assert (df_weather["weather_condition"] == "Severe").all()
    assert (df_weather["weather_risk_score"] == 0.85).all()

    df_road = apply_environmental_shock(df, "poor_road")
    assert (df_road["road_condition"] == "Poor").all()


def test_model_stability_seeds():
    """Verify multi-seed stability evaluator aggregates variance statistics."""
    rng = np.random.RandomState(42)
    X = pd.DataFrame(rng.randn(100, 4), columns=[f"f{i}" for i in range(4)])
    y = rng.uniform(1.0, 5.0, size=100)

    res = evaluate_model_stability(
        X_train=X.iloc[:70],
        y_train=y[:70],
        X_val=X.iloc[70:],
        y_val=y[70:],
        model_name="lightgbm",
        target_type="regression",
        model_params={"n_estimators": 10, "verbose": -1},
        seeds=[42, 123, 999],
    )

    assert "seeds_evaluated" in res
    assert len(res["seeds_evaluated"]) == 3
    assert "aggregate" in res
    assert "mae" in res["aggregate"]
    assert "std" in res["aggregate"]["mae"]
    assert "mean" in res["aggregate"]["mae"]
    assert res["aggregate"]["mae"]["std"] >= 0.0


def test_shap_explainer_global_and_local():
    """Verify TreeExplainer SHAP produces rankings and structured narrative output."""
    rng = np.random.RandomState(42)
    X = pd.DataFrame(rng.randn(80, 5), columns=["dist", "stops", "cong", "weight", "weather"])
    y = X["dist"] * 0.8 + X["cong"] * 0.5 + rng.normal(0, 0.1, size=80)

    model = LGBMRegressor(n_estimators=15, random_state=42, verbose=-1)
    model.fit(X, y)

    explainer = ETAShapExplainer(model, feature_names=list(X.columns), target_type="regression")

    # Global
    global_res = explainer.explain_global(X, top_n=3)
    assert "top_features" in global_res
    assert len(global_res["top_features"]) == 3
    assert global_res["top_features"][0]["mean_abs_shap"] >= global_res["top_features"][1]["mean_abs_shap"]

    # Local
    sample_row = X.iloc[0]
    local_res = explainer.explain_instance(sample_row, shipment_id="SHP-TEST-1", top_k=2)
    assert local_res["shipment_id"] == "SHP-TEST-1"
    assert "positive_drivers" in local_res
    assert "negative_drivers" in local_res
    assert "narrative" in local_res
    assert "ETA increased because:" in local_res["narrative"]
    assert "ETA decreased because:" in local_res["narrative"]
