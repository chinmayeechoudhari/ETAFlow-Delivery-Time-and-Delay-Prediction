"""
ETAFlow Evaluation Package.

Exports subgroup error analysis, robustness testing, and evaluation metrics.
"""

from ml.evaluation.robustness import (
    apply_environmental_shock,
    inject_missingness,
    run_robustness_audit,
)
from ml.evaluation.subgroup import (
    analyze_classification_subgroups,
    analyze_regression_subgroups,
    create_subgroup_slices,
)
from ml.training.evaluation import (
    evaluate_classification,
    evaluate_regression,
)

__all__ = [
    "analyze_classification_subgroups",
    "analyze_regression_subgroups",
    "create_subgroup_slices",
    "inject_missingness",
    "apply_environmental_shock",
    "run_robustness_audit",
    "evaluate_classification",
    "evaluate_regression",
]
