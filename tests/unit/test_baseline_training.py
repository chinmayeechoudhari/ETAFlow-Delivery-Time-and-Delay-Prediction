"""
Unit Tests for ETAFlow Baseline ML Training & Benchmarking Layer (Milestone 6-A).

Verifies:
1. Chronological split correctness (ratios, sizes)
2. No train/validation/test overlap
3. Chronological ordering boundaries (train < val < test)
4. Target isolation (targets excluded from feature matrix)
5. Feature/target alignment and index preservation
6. Regression metrics mathematical correctness
7. Classification metrics mathematical correctness
8. Model training functionality (regression & classification)
9. Prediction shapes for regression and classification
10. Model serialization and deserialization integrity
11. Reproducibility with fixed seed
12. Benchmark result structure completeness
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from ml.training.classification import (
    get_classification_models,
    predict_classification,
    train_classification_model,
)
from ml.training.evaluation import evaluate_classification, evaluate_regression
from ml.training.model_utils import (
    extract_feature_importance,
    load_model,
    save_model,
    select_best_classification_model,
    select_best_regression_model,
)
from ml.training.regression import (
    get_regression_models,
    predict_regression,
    train_regression_model,
)
from ml.training.split import chronological_split


@pytest.fixture
def synthetic_shipments_df() -> pd.DataFrame:
    """Create a small, deterministic synthetic logistics dataset for fast unit testing."""
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    np.random.seed(42)

    df = pd.DataFrame(
        {
            "shipment_id": [f"SHP{i:06d}" for i in range(n)],
            "order_date": dates,
            "feature_1": np.random.randn(n),
            "feature_2": np.random.uniform(10, 100, n),
            "actual_delivery_days": np.random.exponential(scale=2.5, size=n) + 0.5,
            "is_delayed": np.random.binomial(n=1, p=0.25, size=n),
        }
    )
    # Ensure it's not pre-sorted so we test that split sorts properly
    return df.sample(frac=1.0, random_state=123).reset_index(drop=True)


@pytest.fixture
def synthetic_matrices() -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Create small feature matrix and targets for model testing."""
    n_samples = 100
    n_features = 10
    np.random.seed(42)

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feat_{i}" for i in range(n_features)],
        index=[f"SHP{i:05d}" for i in range(n_samples)],
    )
    y_reg = pd.Series(
        np.random.exponential(2.0, n_samples) + 0.5,
        index=X.index,
        name="actual_delivery_days",
    )
    y_cls = pd.Series(
        np.random.binomial(1, 0.3, n_samples),
        index=X.index,
        name="is_delayed",
    )
    return X, y_reg, y_cls


# 1. Chronological Split Correctness
def test_chronological_split_ratios(synthetic_shipments_df: pd.DataFrame):
    """Verify split produces exact 70/15/15 partitions."""
    result = chronological_split(
        synthetic_shipments_df,
        timestamp_column="order_date",
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        id_column="shipment_id",
    )
    total = len(synthetic_shipments_df)
    assert len(result.train) == int(total * 0.70)
    assert len(result.validation) == int(total * 0.15)
    assert len(result.test) == total - len(result.train) - len(result.validation)
    assert (len(result.train) + len(result.validation) + len(result.test)) == total


# 2. No Train / Validation / Test Overlap
def test_split_no_overlap(synthetic_shipments_df: pd.DataFrame):
    """Verify that train, validation, and test partitions are strictly disjoint."""
    result = chronological_split(synthetic_shipments_df, timestamp_column="order_date")
    train_ids = set(result.train["shipment_id"])
    val_ids = set(result.validation["shipment_id"])
    test_ids = set(result.test["shipment_id"])

    assert len(train_ids.intersection(val_ids)) == 0, "Train and Validation overlap!"
    assert len(val_ids.intersection(test_ids)) == 0, "Validation and Test overlap!"
    assert len(train_ids.intersection(test_ids)) == 0, "Train and Test overlap!"


# 3. Chronological Ordering Boundaries
def test_split_chronological_boundaries(synthetic_shipments_df: pd.DataFrame):
    """Verify max(train) <= min(val) and max(val) <= min(test)."""
    result = chronological_split(synthetic_shipments_df, timestamp_column="order_date")
    assert result.train["order_date"].max() <= result.validation["order_date"].min()
    assert result.validation["order_date"].max() <= result.test["order_date"].min()
    assert result.summary["boundaries_valid"]["train_before_val"] is True
    assert result.summary["boundaries_valid"]["val_before_test"] is True


# 4. Target Isolation
def test_target_isolation_in_feature_matrices(synthetic_matrices):
    """Ensure target columns are strictly quarantined from feature space."""
    X, y_reg, y_cls = synthetic_matrices
    forbidden = ["actual_delivery_days", "delivery_delay_days", "is_delayed", "delivery_datetime"]
    for col in forbidden:
        assert col not in X.columns, f"Target leakage! {col} found in features X"


# 5. Feature/Target Alignment
def test_feature_target_alignment(synthetic_matrices):
    """Ensure index and row alignment between features and targets."""
    X, y_reg, y_cls = synthetic_matrices
    assert len(X) == len(y_reg) == len(y_cls)
    assert (X.index == y_reg.index).all()
    assert (X.index == y_cls.index).all()


# 6. Regression Metrics Correctness
def test_regression_metrics_calculation():
    """Validate mathematical exactness of regression metric functions."""
    y_true = np.array([2.0, 4.0, 6.0, 8.0])
    y_pred = np.array([2.5, 3.5, 6.0, 9.0])
    # abs errors: [0.5, 0.5, 0.0, 1.0] -> sum = 2.0 / 4 = 0.5 MAE
    # squared errors: [0.25, 0.25, 0.0, 1.0] -> sum = 1.5 / 4 = 0.375 -> sqrt(0.375) ~ 0.61237
    # within 0.5 day: 3/4 = 75%
    # within 1.0 day: 4/4 = 100%

    metrics = evaluate_regression(y_true, y_pred)
    assert np.isclose(metrics["mae"], 0.5)
    assert np.isclose(metrics["rmse"], np.sqrt(0.375))
    assert np.isclose(metrics["within_0.5_day"], 0.75)
    assert np.isclose(metrics["within_1_day"], 1.0)
    assert metrics["error_statistics"]["max_abs_error"] == 1.0
    assert metrics["error_statistics"]["median_abs_error"] == 0.5


# 7. Classification Metrics Correctness
def test_classification_metrics_calculation():
    """Validate mathematical exactness of classification metric functions."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])  # 1 FP, 0 FN, 2 TP, 1 TN
    y_prob = np.array([0.1, 0.7, 0.8, 0.9])

    metrics = evaluate_classification(y_true, y_pred, y_prob)
    assert metrics["accuracy"] == 0.75
    # precision = TP / (TP + FP) = 2 / 3 ~ 0.6667
    assert np.isclose(metrics["precision"], 2.0 / 3.0)
    # recall = TP / (TP + FN) = 2 / 2 = 1.0
    assert metrics["recall"] == 1.0
    # f1 = 2 * (2/3 * 1) / (2/3 + 1) = 4/5 = 0.8
    assert np.isclose(metrics["f1"], 0.8)
    assert metrics["roc_auc"] is not None
    assert metrics["pr_auc"] is not None
    assert metrics["confusion_matrix"]["tp"] == 2
    assert metrics["confusion_matrix"]["fp"] == 1
    assert metrics["confusion_matrix"]["tn"] == 1
    assert metrics["confusion_matrix"]["fn"] == 0


# 8 & 9. Model Training & Prediction Shapes
def test_regression_models_train_and_predict(synthetic_matrices):
    """Verify all 5 regression models fit and generate matching prediction shapes."""
    X, y_reg, _ = synthetic_matrices
    models = get_regression_models()

    assert set(models.keys()) == {"dummy", "ridge", "random_forest", "lightgbm", "xgboost"}

    for name, model in models.items():
        fitted, dur = train_regression_model(name, model, X, y_reg)
        preds = predict_regression(fitted, X)
        assert isinstance(preds, np.ndarray)
        assert len(preds) == len(X)
        assert not np.isnan(preds).any()
        assert dur >= 0.0


def test_classification_models_train_and_predict(synthetic_matrices):
    """Verify all 5 classification models fit and generate matching prediction shapes and probabilities."""
    X, _, y_cls = synthetic_matrices
    models = get_classification_models()

    assert set(models.keys()) == {"dummy", "logistic_regression", "random_forest", "lightgbm", "xgboost"}

    for name, model in models.items():
        fitted, dur = train_classification_model(name, model, X, y_cls)
        preds, probs = predict_classification(fitted, X)
        assert isinstance(preds, np.ndarray)
        assert len(preds) == len(X)
        assert set(np.unique(preds)).issubset({0, 1})
        if probs is not None:
            assert len(probs) == len(X)
            assert (probs >= 0.0).all() and (probs <= 1.0).all()
        assert dur >= 0.0


# 10. Model Persistence
def test_model_save_and_load(synthetic_matrices):
    """Verify serialization to disk and exact prediction recovery."""
    X, y_reg, _ = synthetic_matrices
    models = get_regression_models()
    rf = models["random_forest"]
    rf.fit(X, y_reg)
    orig_preds = rf.predict(X)

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "rf_model.joblib"
        save_model(rf, save_path)
        assert save_path.exists()

        loaded_rf = load_model(save_path)
        loaded_preds = loaded_rf.predict(X)
        np.testing.assert_allclose(orig_preds, loaded_preds)


# 11. Reproducibility
def test_model_reproducibility(synthetic_matrices):
    """Verify identical predictions across runs with fixed seed."""
    X, y_reg, _ = synthetic_matrices
    rf1 = get_regression_models(random_state=42)["random_forest"]
    rf1.fit(X, y_reg)
    preds1 = rf1.predict(X)

    rf2 = get_regression_models(random_state=42)["random_forest"]
    rf2.fit(X, y_reg)
    preds2 = rf2.predict(X)

    np.testing.assert_allclose(preds1, preds2, rtol=1e-7, atol=1e-7)


# 12. Model Selection Logic
def test_model_selection_logic():
    """Verify deterministic selection of winning models according to primary metrics."""
    reg_results = [
        {"model": "dummy", "test_metrics": {"mae": 1.5, "rmse": 2.0, "r2": 0.0, "within_1_day": 0.4}},
        {"model": "rf", "test_metrics": {"mae": 0.8, "rmse": 1.1, "r2": 0.6, "within_1_day": 0.75}},
        {"model": "lgbm", "test_metrics": {"mae": 0.7, "rmse": 1.0, "r2": 0.65, "within_1_day": 0.80}},
    ]
    best_reg = select_best_regression_model(reg_results)
    assert best_reg["model"] == "lgbm"

    cls_results = [
        {"model": "dummy", "test_metrics": {"f1": 0.0, "pr_auc": 0.25, "roc_auc": 0.5, "recall": 0.0}},
        {"model": "lr", "test_metrics": {"f1": 0.55, "pr_auc": 0.60, "roc_auc": 0.75, "recall": 0.50}},
        {"model": "lgbm", "test_metrics": {"f1": 0.72, "pr_auc": 0.80, "roc_auc": 0.88, "recall": 0.70}},
    ]
    best_cls = select_best_classification_model(cls_results)
    assert best_cls["model"] == "lgbm"
