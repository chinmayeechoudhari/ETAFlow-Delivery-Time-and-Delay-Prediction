"""
ETAFlow Preprocessing and Feature Engineering Package.

Provides modular, reproducible, and leakage-safe components for transforming
raw logistics data into ML-ready feature matrices for ETA and Delay prediction.
"""

from ml.preprocessing.data_loader import (
    DataLoader,
    compute_file_md5,
    verify_dvc_checksum,
)
from ml.preprocessing.feature_engineering import (
    LogisticsFeatureEngineerPointA,
    LogisticsFeatureEngineerPointB,
)
from ml.preprocessing.pipeline import (
    ETAPreprocessingPipeline,
    build_preprocessing_pipeline,
    sanitize_column_name,
)
from ml.preprocessing.transformers import (
    HolidaySanitizer,
    MissingIndicatorAdder,
    TemporalFeatureExtractor,
)
from ml.preprocessing.validation import (
    POINT_B_OPERATIONAL_COLUMNS,
    STRICT_TARGET_COLUMNS,
    DataValidationError,
    PreprocessingValidator,
)

__all__ = [
    "DataLoader",
    "compute_file_md5",
    "verify_dvc_checksum",
    "PreprocessingValidator",
    "DataValidationError",
    "STRICT_TARGET_COLUMNS",
    "POINT_B_OPERATIONAL_COLUMNS",
    "HolidaySanitizer",
    "MissingIndicatorAdder",
    "TemporalFeatureExtractor",
    "LogisticsFeatureEngineerPointA",
    "LogisticsFeatureEngineerPointB",
    "ETAPreprocessingPipeline",
    "build_preprocessing_pipeline",
    "sanitize_column_name",
]
