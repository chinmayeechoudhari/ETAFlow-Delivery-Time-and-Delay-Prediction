"""
ETAFlow Preprocessing Pipeline Module.

Orchestrates raw data cleaning, structural missingness handling,
domain feature engineering, imputation, one-hot encoding, and optional scaling
into a unified, reproducible scikit-learn compatible pipeline.
Strictly isolates targets and prevents data leakage across prediction horizons.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.preprocessing.feature_engineering import (
    LogisticsFeatureEngineerPointA,
    LogisticsFeatureEngineerPointB,
)
from ml.preprocessing.transformers import HolidaySanitizer
from ml.preprocessing.validation import PreprocessingValidator, STRICT_TARGET_COLUMNS

logger = logging.getLogger(__name__)


def sanitize_column_name(name: str) -> str:
    """Sanitize column name for cross-platform and ML framework compatibility.

    Removes special characters, parentheses, brackets, and replaces spaces/slashes
    with underscores.
    """
    clean = re.sub(r"[\[\]<>()]+", "", name)
    clean = re.sub(r"[\s/\\-]+", "_", clean)
    clean = re.sub(r"_+", "_", clean)
    return clean.strip("_")


class ETAPreprocessingPipeline(BaseEstimator, TransformerMixin):
    """End-to-end preprocessing pipeline for ETAFlow delivery prediction."""

    def __init__(
        self,
        config: Dict[str, Any],
        scale_numerical: bool = False,
    ) -> None:
        """Initialize the ETA Preprocessing Pipeline.

        Args:
            config: Feature configuration dictionary loaded from YAML.
            scale_numerical: Whether to apply StandardScaler to numerical features.
        """
        self.config = config
        self.horizon: str = config.get("prediction_horizon", "booking").lower()
        self.id_col: str = config.get("id_column", "shipment_id")
        self.scale_numerical: bool = scale_numerical or config.get("preprocessing", {}).get(
            "scale_numerical", False
        )

        # Preprocessing settings from config
        prep_cfg = config.get("preprocessing", {})
        self.num_strategy: str = prep_cfg.get("numerical_imputer_strategy", "median")
        self.cat_strategy: str = prep_cfg.get("categorical_imputer_strategy", "constant")
        self.cat_fill: str = prep_cfg.get("categorical_fill_value", "Unknown")
        self.holiday_fill: str = prep_cfg.get("holiday_fill_value", "No_Holiday")

        # Column specifications from config
        self.raw_num_cols: List[str] = list(config.get("raw_numerical_features", []))
        self.raw_cat_cols: List[str] = list(config.get("raw_categorical_features", []))
        self.engineered_cols: List[str] = list(config.get("engineered_features", []))

        # Initialize domain feature engineer
        if self.horizon == "booking":
            self.feature_engineer = LogisticsFeatureEngineerPointA()
        elif self.horizon == "dispatch":
            self.feature_engineer = LogisticsFeatureEngineerPointB()
        else:
            raise ValueError(
                f"Unknown prediction horizon '{self.horizon}'. Expected 'booking' or 'dispatch'."
            )

        self.holiday_sanitizer = HolidaySanitizer(fill_value=self.holiday_fill)
        self.column_transformer: Optional[ColumnTransformer] = None
        self.feature_names_out_: Optional[List[str]] = None
        self.is_fitted: bool = False

    def _get_active_columns(self) -> Tuple[List[str], List[str]]:
        """Determine full list of numerical and categorical columns to transform."""
        # All engineered features created in our feature engineers are numerical/binary
        all_num_cols = list(self.raw_num_cols) + list(self.engineered_cols)
        all_cat_cols = list(self.raw_cat_cols)
        return all_num_cols, all_cat_cols

    def extract_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract and isolate target columns and shipment IDs from raw dataset.

        Args:
            df: Input DataFrame containing target columns.

        Returns:
            pd.DataFrame: Targets DataFrame with shipment_id as index.
        """
        targets_df = pd.DataFrame(index=df[self.id_col] if self.id_col in df.columns else df.index)
        targets_df.index.name = self.id_col

        target_mapping = self.config.get("target_columns", {})
        reg_target = target_mapping.get("regression_target", "actual_delivery_days")
        cls_target = target_mapping.get("classification_target", "is_delayed")
        sec_target = target_mapping.get("secondary_target", "delivery_delay_days")

        if reg_target in df.columns:
            targets_df[reg_target] = df[reg_target].values
        if cls_target in df.columns:
            targets_df[cls_target] = df[cls_target].values
        if sec_target in df.columns:
            targets_df[sec_target] = df[sec_target].values
        if "promised_delivery_days" in df.columns:
            targets_df["promised_delivery_days"] = df["promised_delivery_days"].values

        return targets_df

    def fit(self, X: pd.DataFrame, y: Optional[Any] = None) -> ETAPreprocessingPipeline:
        """Fit the preprocessing transformers on training data.

        Args:
            X: Input raw features DataFrame.
            y: Ignored (for scikit-learn API compatibility).

        Returns:
            ETAPreprocessingPipeline: Fitted pipeline instance.
        """
        logger.info("Fitting ETA Preprocessing Pipeline for horizon '%s'...", self.horizon)

        # 1. Step A: Structural holiday sanitization
        X_clean = self.holiday_sanitizer.fit_transform(X)

        # 2. Step B: Domain feature engineering
        X_eng = self.feature_engineer.fit_transform(X_clean)

        # 3. Step C: Build ColumnTransformer
        num_cols, cat_cols = self._get_active_columns()

        # Validate that required columns are present in engineered dataframe
        missing_num = [c for c in num_cols if c not in X_eng.columns]
        missing_cat = [c for c in cat_cols if c not in X_eng.columns]
        if missing_num or missing_cat:
            raise KeyError(
                f"Missing columns in engineered dataframe: numerical={missing_num}, categorical={missing_cat}"
            )

        # Numerical pipeline
        num_steps = [("imputer", SimpleImputer(strategy=self.num_strategy))]
        if self.scale_numerical:
            num_steps.append(("scaler", StandardScaler()))
        num_pipeline = Pipeline(steps=num_steps)

        # Categorical pipeline
        ohe_cfg = self.config.get("preprocessing", {}).get("one_hot_encoding", {})
        cat_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy=self.cat_strategy, fill_value=self.cat_fill)),
                (
                    "ohe",
                    OneHotEncoder(
                        handle_unknown=ohe_cfg.get("handle_unknown", "ignore"),
                        sparse_output=False,
                    ),
                ),
            ]
        )

        self.column_transformer = ColumnTransformer(
            transformers=[
                ("num", num_pipeline, num_cols),
                ("cat", cat_pipeline, cat_cols),
            ],
            remainder="drop",
            verbose_feature_names_out=True,
        )
        self.column_transformer.set_output(transform="pandas")

        # Fit column transformer
        self.column_transformer.fit(X_eng)

        # Extract and clean feature names
        raw_names = self.column_transformer.get_feature_names_out()
        cleaned_names = []
        for name in raw_names:
            clean = name.replace("num__", "").replace("cat__", "")
            clean = sanitize_column_name(clean)
            cleaned_names.append(clean)
        self.feature_names_out_ = cleaned_names

        self.is_fitted = True
        logger.info(
            "Pipeline successfully fitted for '%s' (%d features generated).",
            self.horizon,
            len(self.feature_names_out_),
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform input data into an ML-ready feature matrix.

        Args:
            X: Input DataFrame.

        Returns:
            pd.DataFrame: ML-ready feature matrix with shipment_id as index.
        """
        if not self.is_fitted or self.column_transformer is None:
            raise RuntimeError("Pipeline must be fitted before calling transform().")

        # Preserve shipment_id for indexing
        shipment_ids = None
        if self.id_col in X.columns:
            shipment_ids = X[self.id_col].values
        elif X.index.name == self.id_col:
            shipment_ids = X.index.values

        # 1. Step A: Structural holiday sanitization
        X_clean = self.holiday_sanitizer.transform(X)

        # 2. Step B: Domain feature engineering
        X_eng = self.feature_engineer.transform(X_clean)

        # 3. Step C: Column transformer
        transformed_df = self.column_transformer.transform(X_eng)

        # 4. Step D: Assign sanitized column names
        transformed_df.columns = self.feature_names_out_

        # 5. Step E: Restore shipment_id index
        if shipment_ids is not None:
            transformed_df.index = pd.Index(shipment_ids, name=self.id_col)

        # 6. Step F: Strict feature isolation validation
        PreprocessingValidator.validate_feature_isolation(
            transformed_df.columns.tolist(), self.horizon
        )

        return transformed_df

    def fit_transform(self, X: pd.DataFrame, y: Optional[Any] = None) -> pd.DataFrame:
        """Fit and transform in a single pass.

        Args:
            X: Input DataFrame.
            y: Ignored.

        Returns:
            pd.DataFrame: Processed ML feature matrix.
        """
        return self.fit(X, y).transform(X)

    def get_feature_names(self) -> List[str]:
        """Return the list of generated ML feature column names."""
        if not self.is_fitted or self.feature_names_out_ is None:
            raise RuntimeError("Pipeline must be fitted to access feature names.")
        return list(self.feature_names_out_)

    def save(self, filepath: Union[str, Path]) -> None:
        """Serialize the fitted pipeline to disk using joblib.

        Args:
            filepath: Destination file path.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Pipeline serialized to %s", path)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> ETAPreprocessingPipeline:
        """Load a serialized pipeline from disk.

        Args:
            filepath: Source file path.

        Returns:
            ETAPreprocessingPipeline: Deserialized pipeline instance.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Pipeline file not found: {path}")
        pipeline = joblib.load(path)
        logger.info("Pipeline loaded successfully from %s", path)
        return pipeline


def build_preprocessing_pipeline(
    config_path: Union[str, Path],
    scale_numerical: bool = False,
) -> ETAPreprocessingPipeline:
    """Factory function to instantiate ETAPreprocessingPipeline from a config file.

    Args:
        config_path: Path to features YAML configuration.
        scale_numerical: Whether to enable feature scaling.

    Returns:
        ETAPreprocessingPipeline: Configured pipeline instance.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return ETAPreprocessingPipeline(config=config, scale_numerical=scale_numerical)
