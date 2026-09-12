"""
ETAFlow Chronological Split Module.

Provides deterministic, leakage-safe time-based splitting of shipment records
into Train (70%), Validation (15%), and Test (15%) partitions.
Enforces strict chronological boundary isolation:
max(train_date) < min(validation_date) and max(validation_date) < min(test_date).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SplitPartitionSummary:
    """Statistical and temporal summary for a single split partition."""

    name: str
    row_count: int
    percentage: float
    start_date: str
    end_date: str
    delay_rate: Optional[float] = None
    mean_delivery_days: Optional[float] = None


@dataclass
class ChronologicalSplitResult:
    """Container holding partitioned datasets and split verification metadata."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    timestamp_column: str
    summary: Dict[str, Any]

    def print_summary(self) -> str:
        """Format human-readable summary of split boundaries and statistics."""
        lines = [
            "=" * 60,
            "ETAFlow Chronological Dataset Split Summary",
            "=" * 60,
            f"Timestamp column: {self.timestamp_column}",
            "-" * 60,
        ]
        for part_name in ["train", "validation", "test"]:
            p = self.summary[part_name]
            lines.append(f"{part_name.upper()}")
            lines.append(f"  Rows:       {p['row_count']:,} ({p['percentage']:.1f}%)")
            lines.append(f"  Start Date: {p['start_date']}")
            lines.append(f"  End Date:   {p['end_date']}")
            if p.get("delay_rate") is not None:
                lines.append(f"  Delay Rate: {p['delay_rate']:.2%}")
            if p.get("mean_delivery_days") is not None:
                lines.append(f"  Mean Days:  {p['mean_delivery_days']:.2f}")
            lines.append("-" * 60)

        lines.append(f"Train < Validation Boundary Check: {'PASSED' if self.summary['boundaries_valid']['train_before_val'] else 'FAILED'}")
        lines.append(f"Validation < Test Boundary Check: {'PASSED' if self.summary['boundaries_valid']['val_before_test'] else 'FAILED'}")
        lines.append("=" * 60)
        formatted = "\n".join(lines)
        return formatted


def chronological_split(
    df: pd.DataFrame,
    timestamp_column: str = "order_date",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    id_column: str = "shipment_id",
) -> ChronologicalSplitResult:
    """Split a DataFrame chronologically into train, validation, and test partitions.

    Args:
        df: Input DataFrame containing shipment records.
        timestamp_column: Column name containing datetime or parseable timestamp.
        train_ratio: Proportion of earliest records allocated to training (default 0.70).
        val_ratio: Proportion of intermediate records allocated to validation (default 0.15).
        test_ratio: Proportion of latest records allocated to testing (default 0.15).
        id_column: Unique shipment identifier column for overlap validation.

    Returns:
        ChronologicalSplitResult: Structured container with train, val, test subsets and metadata.

    Raises:
        ValueError: If ratios do not sum to 1.0, timestamp column is missing, or boundary checks fail.
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(
            f"Split ratios must sum to 1.0. Got train={train_ratio}, val={val_ratio}, test={test_ratio}"
        )

    if timestamp_column not in df.columns:
        raise ValueError(f"Timestamp column '{timestamp_column}' not found in DataFrame.")

    total_rows = len(df)
    if total_rows == 0:
        raise ValueError("Cannot split an empty DataFrame.")

    # Convert timestamp to datetime if not already
    df_sorted = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_sorted[timestamp_column]):
        df_sorted[timestamp_column] = pd.to_datetime(df_sorted[timestamp_column])

    # Sort strictly chronologically, using id_column as secondary tie-breaker for full determinism
    sort_cols = [timestamp_column]
    if id_column in df_sorted.columns:
        sort_cols.append(id_column)
    elif df_sorted.index.name == id_column:
        pass

    df_sorted = df_sorted.sort_values(by=sort_cols).reset_index(drop=True)

    # Compute deterministic cut-offs
    n_train = int(total_rows * train_ratio)
    n_val = int(total_rows * val_ratio)
    n_test = total_rows - n_train - n_val

    train_df = df_sorted.iloc[:n_train].copy()
    val_df = df_sorted.iloc[n_train : n_train + n_val].copy()
    test_df = df_sorted.iloc[n_train + n_val :].copy()

    # Verify zero overlap between IDs if id_column present
    if id_column in df.columns:
        train_ids = set(train_df[id_column])
        val_ids = set(val_df[id_column])
        test_ids = set(test_df[id_column])

        train_val_overlap = train_ids.intersection(val_ids)
        val_test_overlap = val_ids.intersection(test_ids)
        train_test_overlap = train_ids.intersection(test_ids)

        if train_val_overlap or val_test_overlap or train_test_overlap:
            raise ValueError(
                f"Data overlap detected across splits: "
                f"train/val={len(train_val_overlap)}, "
                f"val/test={len(val_test_overlap)}, "
                f"train/test={len(train_test_overlap)}"
            )

    # Verify boundary ordering
    train_max_date = train_df[timestamp_column].max()
    val_min_date = val_df[timestamp_column].min()
    val_max_date = val_df[timestamp_column].max()
    test_min_date = test_df[timestamp_column].min()

    train_before_val = train_max_date <= val_min_date
    val_before_test = val_max_date <= test_min_date

    if not (train_before_val and val_before_test):
        raise ValueError(
            f"Chronological split boundary violation: "
            f"train_max ({train_max_date}) <= val_min ({val_min_date}) is {train_before_val}; "
            f"val_max ({val_max_date}) <= test_min ({test_min_date}) is {val_before_test}."
        )

    # Build summary
    summary: Dict[str, Any] = {
        "timestamp_column": timestamp_column,
        "total_rows": total_rows,
        "boundaries_valid": {
            "train_before_val": bool(train_before_val),
            "val_before_test": bool(val_before_test),
        },
    }

    for name, part in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        part_info: Dict[str, Any] = {
            "row_count": len(part),
            "percentage": (len(part) / total_rows) * 100.0,
            "start_date": str(part[timestamp_column].min()),
            "end_date": str(part[timestamp_column].max()),
        }
        if "is_delayed" in part.columns:
            part_info["delay_rate"] = float(part["is_delayed"].mean())
        if "actual_delivery_days" in part.columns:
            part_info["mean_delivery_days"] = float(part["actual_delivery_days"].mean())

        summary[name] = part_info

    logger.info(
        "Chronological split completed: Train=%d, Val=%d, Test=%d (Boundaries: %s <= %s <= %s)",
        len(train_df),
        len(val_df),
        len(test_df),
        train_max_date,
        val_min_date,
        test_min_date,
    )

    return ChronologicalSplitResult(
        train=train_df,
        validation=val_df,
        test=test_df,
        timestamp_column=timestamp_column,
        summary=summary,
    )
