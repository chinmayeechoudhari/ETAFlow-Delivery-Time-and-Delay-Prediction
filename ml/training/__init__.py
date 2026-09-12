"""
ETAFlow Training and Benchmarking Package.

Exports core functions and classes for model definition, chronological splitting,
leakage-free training, and standardized evaluation.
"""

from ml.training.classification import (
    get_classification_models,
    predict_classification,
    train_classification_model,
)
from ml.training.evaluation import (
    evaluate_classification,
    evaluate_regression,
)
from ml.training.model_utils import (
    extract_feature_importance,
    load_model,
    measure_execution_time,
    save_model,
    select_best_classification_model,
    select_best_regression_model,
)
from ml.training.regression import (
    get_regression_models,
    predict_regression,
    train_regression_model,
)
from ml.training.split import (
    ChronologicalSplitResult,
    chronological_split,
)
from ml.training.train import (
    run_baseline_benchmark,
)

__all__ = [
    "ChronologicalSplitResult",
    "chronological_split",
    "get_regression_models",
    "train_regression_model",
    "predict_regression",
    "get_classification_models",
    "train_classification_model",
    "predict_classification",
    "evaluate_regression",
    "evaluate_classification",
    "measure_execution_time",
    "save_model",
    "load_model",
    "extract_feature_importance",
    "select_best_regression_model",
    "select_best_classification_model",
    "run_baseline_benchmark",
]
