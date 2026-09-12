"""
ETAFlow Sliced Error and Subgroup Analysis Module.

Audits model performance across operational subgroups (carriers, transport modes,
geographies, distance tiers, weight classes, weather, traffic, and holidays)
to surface hidden vulnerabilities and failure modes.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, mean_absolute_error, precision_score, recall_score, root_mean_squared_error

logger = logging.getLogger(__name__)


def create_subgroup_slices(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich dataframe with standardized categorical bins for subgroup slicing."""
    df_slices = df.copy()

    # Continuous binning: Distance
    if "distance_km" in df_slices.columns:
        df_slices["distance_tier"] = pd.cut(
            df_slices["distance_km"],
            bins=[0, 500, 1500, 3000, float("inf")],
            labels=["Short (<500km)", "Medium (500-1500km)", "Long (1500-3000km)", "Ultra-Long (>3000km)"],
            right=True,
        ).astype(str)

    # Continuous binning: Package Weight
    if "package_weight_kg" in df_slices.columns:
        df_slices["weight_tier"] = pd.cut(
            df_slices["package_weight_kg"],
            bins=[0, 5, 20, 50, float("inf")],
            labels=["Small (<5kg)", "Medium (5-20kg)", "Heavy (20-50kg)", "Freight (>50kg)"],
            right=True,
        ).astype(str)

    # Season extraction from order_date if present
    if "order_date" in df_slices.columns:
        dates = pd.to_datetime(df_slices["order_date"])
        month = dates.dt.month
        df_slices["season"] = month.map(
            lambda m: "Winter" if m in [12, 1, 2] else ("Spring" if m in [3, 4, 5] else ("Summer" if m in [6, 7, 8] else "Fall"))
        )

    return df_slices


def analyze_regression_subgroups(
    df_raw: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    slice_columns: Optional[List[str]] = None,
    min_samples: int = 30,
) -> Dict[str, List[Dict[str, Any]]]:
    """Perform sliced regression error analysis across subgroup features.

    Args:
        df_raw: Original raw metadata DataFrame corresponding to evaluated samples.
        y_true: Actual delivery days array.
        y_pred: Predicted delivery days array.
        slice_columns: List of columns to slice by.
        min_samples: Minimum sample threshold to report meaningful findings.

    Returns:
        Dict[str, List[Dict[str, Any]]]: Mapping from slice category to list of subgroup metrics.
    """
    df_enriched = create_subgroup_slices(df_raw).reset_index(drop=True)
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    residuals = y_p - y_t

    if slice_columns is None:
        slice_columns = [
            "transport_mode",
            "carrier",
            "origin_hub",
            "destination_hub",
            "distance_tier",
            "weight_tier",
            "weather_condition",
            "traffic_level",
            "road_condition",
            "is_holiday",
            "season",
        ]

    results: Dict[str, List[Dict[str, Any]]] = {}

    for col in slice_columns:
        if col not in df_enriched.columns:
            continue

        col_results: List[Dict[str, Any]] = []
        for group_val, group_idx in df_enriched.groupby(col).groups.items():
            idx = np.asarray(group_idx)
            count = len(idx)
            if count < min_samples:
                continue

            sub_yt = y_t[idx]
            sub_yp = y_p[idx]
            sub_res = residuals[idx]

            mae = float(mean_absolute_error(sub_yt, sub_yp))
            rmse = float(root_mean_squared_error(sub_yt, sub_yp))
            mean_bias = float(np.mean(sub_res))  # positive = overpredicting, negative = underpredicting

            col_results.append({
                "subgroup": str(group_val),
                "sample_count": count,
                "sample_percentage": round(count / len(y_t) * 100, 2),
                "mae": round(mae, 4),
                "rmse": round(rmse, 4),
                "mean_bias": round(mean_bias, 4),
                "bias_direction": "Overprediction" if mean_bias > 0.05 else ("Underprediction" if mean_bias < -0.05 else "Neutral"),
            })

        # Sort worst MAE first
        col_results.sort(key=lambda x: x["mae"], reverse=True)
        results[col] = col_results

    return results


def analyze_classification_subgroups(
    df_raw: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    slice_columns: Optional[List[str]] = None,
    min_samples: int = 30,
) -> Dict[str, List[Dict[str, Any]]]:
    """Perform sliced classification error analysis across subgroup features.

    Args:
        df_raw: Original raw metadata DataFrame corresponding to evaluated samples.
        y_true: Actual binary delay array (0 or 1).
        y_pred: Predicted binary delay array (0 or 1).
        slice_columns: List of columns to slice by.
        min_samples: Minimum sample threshold to report meaningful findings.

    Returns:
        Dict[str, List[Dict[str, Any]]]: Mapping from slice category to list of subgroup metrics.
    """
    df_enriched = create_subgroup_slices(df_raw).reset_index(drop=True)
    y_t = np.asarray(y_true, dtype=np.int64)
    y_p = np.asarray(y_pred, dtype=np.int64)

    if slice_columns is None:
        slice_columns = [
            "transport_mode",
            "carrier",
            "origin_hub",
            "destination_hub",
            "distance_tier",
            "weight_tier",
            "weather_condition",
            "traffic_level",
            "road_condition",
            "is_holiday",
            "season",
        ]

    results: Dict[str, List[Dict[str, Any]]] = {}

    for col in slice_columns:
        if col not in df_enriched.columns:
            continue

        col_results: List[Dict[str, Any]] = []
        for group_val, group_idx in df_enriched.groupby(col).groups.items():
            idx = np.asarray(group_idx)
            count = len(idx)
            if count < min_samples:
                continue

            sub_yt = y_t[idx]
            sub_yp = y_p[idx]

            prec = float(precision_score(sub_yt, sub_yp, zero_division=0))
            rec = float(recall_score(sub_yt, sub_yp, zero_division=0))
            f1 = float(f1_score(sub_yt, sub_yp, zero_division=0))

            cm = confusion_matrix(sub_yt, sub_yp, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()

            fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
            fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

            col_results.append({
                "subgroup": str(group_val),
                "sample_count": count,
                "sample_percentage": round(count / len(y_t) * 100, 2),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "fpr": round(fpr, 4),
                "fnr": round(fnr, 4),
            })

        # Sort lowest F1 first (worst performance)
        col_results.sort(key=lambda x: x["f1"])
        results[col] = col_results

    return results
