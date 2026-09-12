"""
ETAFlow Milestone 6-B Advanced Optimization Runner Script.

CLI entrypoint to execute end-to-end model optimization, validation, calibration,
error analysis, robustness stress testing, multi-seed stability, and SHAP explainability.

Usage:
    python scripts/run_optimization.py --all
    python scripts/run_optimization.py --all --trials 20
    python scripts/run_optimization.py --phase cv
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure repository root is on sys.path
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from ml.preprocessing.pipeline import ETAPreprocessingPipeline, build_preprocessing_pipeline
from ml.training.classification import LGBMClassifier, XGBClassifier
from ml.training.optimize import ETAOptimizationPipeline
from ml.training.regression import LGBMRegressor, XGBRegressor
from ml.training.time_cv import instantiate_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_optimization")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ETAFlow Milestone 6-B Optimization Pipeline Runner"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model_optimization.yaml",
        help="Path to optimization configuration file.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Execute all optimization and evaluation phases sequentially.",
    )
    parser.add_argument(
        "--phase",
        type=str,
        choices=[
            "cv",
            "hyperopt",
            "threshold",
            "calibration",
            "subgroup",
            "robustness",
            "stability",
            "shap",
            "selection",
        ],
        help="Execute a single specific phase.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Override number of Optuna trials per model.",
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Limit number of dataset rows loaded (for rapid testing).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_total = time.time()

    pipeline = ETAOptimizationPipeline(config_path=args.config)

    logger.info("Initializing ETAFlow M6-B Advanced Optimization Pipeline...")
    train_df, val_df, test_df = pipeline.load_and_split_data(nrows=args.nrows)

    # Preprocess Train and Val partitions for Booking and Dispatch
    logger.info("Fitting Booking Preprocessing Pipeline on Train...")
    pipe_booking = build_preprocessing_pipeline("configs/features_booking.yaml")
    X_train_book = pipe_booking.fit_transform(train_df)
    X_val_book = pipe_booking.transform(val_df)

    logger.info("Fitting Dispatch Preprocessing Pipeline on Train...")
    pipe_dispatch = build_preprocessing_pipeline("configs/features_dispatch.yaml")
    X_train_disp = pipe_dispatch.fit_transform(train_df)
    X_val_disp = pipe_dispatch.transform(val_df)

    pipelines_dict = {
        "booking": pipe_booking,
        "dispatch": pipe_dispatch,
    }

    X_train_dict = {
        "booking": X_train_book,
        "dispatch": X_train_disp,
    }
    X_val_dict = {
        "booking": X_val_book,
        "dispatch": X_val_disp,
    }

    y_train_dict = {
        "booking": {
            "regression": train_df["actual_delivery_days"].values,
            "classification": train_df["is_delayed"].values,
        },
        "dispatch": {
            "regression": train_df["actual_delivery_days"].values,
            "classification": train_df["is_delayed"].values,
        },
    }
    y_val_dict = {
        "booking": {
            "regression": val_df["actual_delivery_days"].values,
            "classification": val_df["is_delayed"].values,
        },
        "dispatch": {
            "regression": val_df["actual_delivery_days"].values,
            "classification": val_df["is_delayed"].values,
        },
    }

    # Phase 2: Chronological Expanding-Window Cross-Validation
    cv_results = pipeline.run_phase_cv(train_df)

    # Phase 3: Hyperparameter Optimization
    opt_results = pipeline.run_phase_hyperopt(
        X_train_dict=X_train_dict,
        y_train_dict=y_train_dict,
        X_val_dict=X_val_dict,
        y_val_dict=y_val_dict,
        n_trials=args.trials,
    )

    # Instantiate and fit optimized models
    logger.info("Fitting optimized champion models on full Train partition...")
    models_dict = {
        "booking_regression_lightgbm": instantiate_model(
            "lightgbm", "regression", opt_results["booking_regression_lightgbm"]["best_params"]
        ),
        "booking_classification_xgboost": instantiate_model(
            "xgboost", "classification", opt_results["booking_classification_xgboost"]["best_params"]
        ),
        "dispatch_regression_lightgbm": instantiate_model(
            "lightgbm", "regression", opt_results["dispatch_regression_lightgbm"]["best_params"]
        ),
        "dispatch_classification_lightgbm": instantiate_model(
            "lightgbm", "classification", opt_results["dispatch_classification_lightgbm"]["best_params"]
        ),
    }

    models_dict["booking_regression_lightgbm"].fit(
        X_train_book, y_train_dict["booking"]["regression"]
    )
    models_dict["booking_classification_xgboost"].fit(
        X_train_book, y_train_dict["booking"]["classification"]
    )
    models_dict["dispatch_regression_lightgbm"].fit(
        X_train_disp, y_train_dict["dispatch"]["regression"]
    )
    models_dict["dispatch_classification_lightgbm"].fit(
        X_train_disp, y_train_dict["dispatch"]["classification"]
    )

    # Phase 4: Threshold Optimization
    threshold_results = pipeline.run_phase_threshold_tuning(
        models_dict=models_dict,
        X_val_dict=X_val_dict,
        y_val_dict=y_val_dict,
    )
    thresholds = {
        "booking": threshold_results["booking"]["optimal_threshold"],
        "dispatch": threshold_results["dispatch"]["optimal_threshold"],
    }

    # Phase 5: Calibration
    cal_results = pipeline.run_phase_calibration(
        models_dict=models_dict,
        X_train_dict=X_train_dict,
        y_train_dict=y_train_dict,
        X_val_dict=X_val_dict,
        y_val_dict=y_val_dict,
        thresholds=thresholds,
    )

    # Phase 6: Subgroup Analysis
    subgroup_results = pipeline.run_phase_subgroup(
        models_dict=models_dict,
        val_df=val_df,
        X_val_dict=X_val_dict,
        thresholds=thresholds,
    )

    # Phase 7: Robustness Stress Testing
    rob_results = pipeline.run_phase_robustness(
        models_dict=models_dict,
        pipelines_dict=pipelines_dict,
        val_df=val_df,
        thresholds=thresholds,
    )

    # Phase 8: Multi-Seed Stability
    stability_results = pipeline.run_phase_stability(
        X_train_dict=X_train_dict,
        y_train_dict=y_train_dict,
        X_val_dict=X_val_dict,
        y_val_dict=y_val_dict,
        opt_results=opt_results,
        thresholds=thresholds,
    )

    # Phase 9: SHAP Explainability
    shap_results = pipeline.run_phase_shap(
        models_dict=models_dict,
        pipelines_dict=pipelines_dict,
        val_df=val_df,
        X_val_dict=X_val_dict,
    )

    # Phase 10: Final Candidate Selection & Quarantined Test Evaluation
    selection_summary = pipeline.run_phase_selection_and_test(
        models_dict=models_dict,
        pipelines_dict=pipelines_dict,
        test_df=test_df,
        opt_results=opt_results,
        cv_results=cv_results,
        thresholds=thresholds,
        cal_results=cal_results,
        stability_results=stability_results,
        rob_results=rob_results,
    )

    total_time = time.time() - start_total
    logger.info("================================================================================")
    logger.info("ETAFlow Milestone 6-B Advanced Optimization Complete in %.1f seconds!", total_time)
    logger.info("Saved reports across all reports/ subdirectories.")
    logger.info("Saved serialized models to models/optimized/")
    logger.info("================================================================================")


if __name__ == "__main__":
    main()
