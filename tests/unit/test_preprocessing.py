"""
Unit Tests for ETAFlow Preprocessing and Feature Engineering Layer.

Verifies:
1. Raw dataset immutability and checksum preservation
2. Target isolation and zero target leakage
3. Horizon separation (Point A booking vs. Point B dispatch lookahead exclusion)
4. Missing value imputation and zero remaining NaNs
5. One-hot encoding and unseen category handling
6. Domain feature engineering mathematical correctness
7. Shipment ID traceability and row count preservation
8. Pipeline reproducibility and determinism
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import yaml

from ml.preprocessing import (
    DataLoader,
    HolidaySanitizer,
    LogisticsFeatureEngineerPointA,
    LogisticsFeatureEngineerPointB,
    MissingIndicatorAdder,
    POINT_B_OPERATIONAL_COLUMNS,
    STRICT_TARGET_COLUMNS,
    TemporalFeatureExtractor,
    build_preprocessing_pipeline,
    compute_file_md5,
    verify_dvc_checksum,
)

RAW_DATA_PATH = Path("data/raw/shipments.csv")
CONFIG_DIR = Path("configs")


@pytest.fixture(scope="module")
def sample_raw_df() -> pd.DataFrame:
    """Load a modest sample of raw records for fast unit testing."""
    loader = DataLoader(RAW_DATA_PATH, verify_checksum=False)
    return loader.load_data(nrows=1000)


def test_raw_data_untouched():
    """Verify that data/raw/shipments.csv MD5 checksum matches DVC specification exactly."""
    assert RAW_DATA_PATH.exists(), "Raw dataset file missing!"
    dvc_file = Path(f"{RAW_DATA_PATH}.dvc")
    assert dvc_file.exists(), "DVC tracking file missing!"

    with open(dvc_file, "r", encoding="utf-8") as f:
        dvc_meta = yaml.safe_load(f)
    expected_md5 = dvc_meta["outs"][0]["md5"]

    actual_md5 = compute_file_md5(RAW_DATA_PATH)
    assert actual_md5.lower() == expected_md5.lower(), (
        f"RAW DATASET MODIFIED! Expected {expected_md5}, got {actual_md5}"
    )


def test_target_isolation(sample_raw_df: pd.DataFrame):
    """Ensure target columns are completely absent from both feature matrices."""
    pipe_a = build_preprocessing_pipeline(CONFIG_DIR / "features_booking.yaml")
    X_a = pipe_a.fit_transform(sample_raw_df)

    for target_col in STRICT_TARGET_COLUMNS:
        assert target_col not in X_a.columns, f"Target leakage! '{target_col}' found in Point A features"

    pipe_b = build_preprocessing_pipeline(CONFIG_DIR / "features_dispatch.yaml")
    X_b = pipe_b.fit_transform(sample_raw_df)

    for target_col in STRICT_TARGET_COLUMNS:
        assert target_col not in X_b.columns, f"Target leakage! '{target_col}' found in Point B features"


def test_leakage_exclusion_point_a(sample_raw_df: pd.DataFrame):
    """Ensure Point B operational metrics are strictly excluded from Point A features."""
    pipe_a = build_preprocessing_pipeline(CONFIG_DIR / "features_booking.yaml")
    X_a = pipe_a.fit_transform(sample_raw_df)

    for op_col in POINT_B_OPERATIONAL_COLUMNS:
        assert op_col not in X_a.columns, (
            f"Lookahead leakage! Operational column '{op_col}' found in Point A features"
        )


def test_missing_values_handled(sample_raw_df: pd.DataFrame):
    """Verify that all missing values are imputed and zero NaNs remain."""
    pipe_a = build_preprocessing_pipeline(CONFIG_DIR / "features_booking.yaml")
    X_a = pipe_a.fit_transform(sample_raw_df)
    assert X_a.isna().sum().sum() == 0, "Point A feature matrix contains unhandled NaNs!"

    pipe_b = build_preprocessing_pipeline(CONFIG_DIR / "features_dispatch.yaml")
    X_b = pipe_b.fit_transform(sample_raw_df)
    assert X_b.isna().sum().sum() == 0, "Point B feature matrix contains unhandled NaNs!"


def test_structural_holiday_handling():
    """Verify that HolidaySanitizer replaces NaN with 'No_Holiday'."""
    df = pd.DataFrame({"holiday_name": [np.nan, "Diwali Festival Week", None]})
    sanitizer = HolidaySanitizer()
    out = sanitizer.transform(df)
    assert (out["holiday_name"] == ["No_Holiday", "Diwali Festival Week", "No_Holiday"]).all()


def test_one_hot_unknown_handling(sample_raw_df: pd.DataFrame):
    """Ensure that unseen categories during inference are safely ignored without raising errors."""
    pipe = build_preprocessing_pipeline(CONFIG_DIR / "features_booking.yaml")
    pipe.fit(sample_raw_df)

    # Create unseen category data
    unseen_df = sample_raw_df.iloc[:10].copy()
    unseen_df["origin_city"] = "UnknownCity999"
    unseen_df["carrier"] = "UnknownCarrier999"
    unseen_df["vehicle_type"] = "FlyingDrone"

    X_unseen = pipe.transform(unseen_df)
    assert len(X_unseen) == 10
    assert X_unseen.isna().sum().sum() == 0
    assert len(X_unseen.columns) == len(pipe.get_feature_names())


def test_feature_engineering_math():
    """Validate mathematical correctness of engineered domain features."""
    sample = pd.DataFrame(
        {
            "distance_km": [100.0, 500.0],
            "package_weight_kg": [2.0, 10.0],
            "package_volume_cm3": [8000.0, 50000.0],  # 0.008 m3, 0.05 m3
            "shipment_value_inr": [1000.0, 5000.0],
            "origin_state": ["Maharashtra", "Maharashtra"],
            "destination_state": ["Maharashtra", "Karnataka"],
            "order_date": ["2024-01-01 10:00:00", "2024-01-01 15:00:00"],
            "estimated_dispatch_datetime": ["2024-01-01 18:00:00", "2024-01-02 03:00:00"],
        }
    )
    fe_a = LogisticsFeatureEngineerPointA()
    out_a = fe_a.transform(sample)

    # Check log transforms
    assert np.isclose(out_a["log_distance_km"].iloc[0], np.log1p(100.0))
    assert np.isclose(out_a["log_package_weight"].iloc[0], np.log1p(2.0))

    # Check density: 2.0 kg / 0.008 m3 = 250.0 kg/m3
    assert np.isclose(out_a["package_density_kg_m3"].iloc[0], 250.0)

    # Check value per kg: 1000 / 2 = 500.0
    assert np.isclose(out_a["value_per_kg"].iloc[0], 500.0, atol=1e-1)

    # Check is_same_state
    assert out_a["is_same_state"].iloc[0] == 1
    assert out_a["is_same_state"].iloc[1] == 0

    # Check estimated dispatch lead hours: (18 - 10) = 8.0 hours
    assert np.isclose(out_a["estimated_dispatch_lead_hours"].iloc[0], 8.0)


def test_target_extraction(sample_raw_df: pd.DataFrame):
    """Verify target extraction extracts correct target columns and preserves IDs."""
    pipe = build_preprocessing_pipeline(CONFIG_DIR / "features_booking.yaml")
    targets_df = pipe.extract_targets(sample_raw_df)

    assert "actual_delivery_days" in targets_df.columns
    assert "is_delayed" in targets_df.columns
    assert "delivery_delay_days" in targets_df.columns
    assert len(targets_df) == len(sample_raw_df)
    assert (targets_df.index == sample_raw_df["shipment_id"]).all()

    # Targets validity
    assert (targets_df["actual_delivery_days"] > 0).all()
    assert set(targets_df["is_delayed"].unique()).issubset({0, 1})
    assert (targets_df["delivery_delay_days"] >= 0).all()


def test_booking_vs_dispatch_separation(sample_raw_df: pd.DataFrame):
    """Verify that Booking and Dispatch pipelines produce distinct feature spaces."""
    pipe_a = build_preprocessing_pipeline(CONFIG_DIR / "features_booking.yaml")
    pipe_b = build_preprocessing_pipeline(CONFIG_DIR / "features_dispatch.yaml")

    X_a = pipe_a.fit_transform(sample_raw_df)
    X_b = pipe_b.fit_transform(sample_raw_df)

    # Point B must contain strictly more features than Point A (operational durations, telematics)
    assert len(X_b.columns) > len(X_a.columns), "Point B should contain more features than Point A"

    # Specific Point B telematics must not be in Point A
    b_only_features = [
        "traffic_weather_interaction",
        "order_to_pickup_hours",
        "pickup_to_dispatch_hours",
        "dispatch_delay_hours",
        "stops_per_100km",
        "complexity_per_stop",
    ]
    for b_feat in b_only_features:
        assert b_feat in X_b.columns, f"Expected '{b_feat}' in Point B features"
        assert b_feat not in X_a.columns, f"Unexpected '{b_feat}' found in Point A features"


def test_reproducibility(sample_raw_df: pd.DataFrame):
    """Verify pipeline produces deterministic, bitwise identical output on repeated runs."""
    pipe1 = build_preprocessing_pipeline(CONFIG_DIR / "features_booking.yaml")
    X1 = pipe1.fit_transform(sample_raw_df)

    pipe2 = build_preprocessing_pipeline(CONFIG_DIR / "features_booking.yaml")
    X2 = pipe2.fit_transform(sample_raw_df)

    pd.testing.assert_frame_equal(X1, X2)
