#!/usr/bin/env python3
"""
ETAFlow Baseline Model Training & Benchmarking CLI (Milestone 6-A).

Executes reproducible baseline model benchmarking for delivery-time regression
and delay risk classification across Booking and Dispatch horizons.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure repository root is on sys.path
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from ml.training.train import run_baseline_benchmark

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train_baseline")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="ETAFlow Baseline ML Benchmarking CLI (Milestone 6-A)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model_baseline.yaml",
        help="Path to baseline model configuration YAML (default: configs/model_baseline.yaml)",
    )
    parser.add_argument(
        "--horizon",
        type=str,
        choices=["booking", "dispatch", "all"],
        default="all",
        help="Prediction horizon to benchmark: 'booking', 'dispatch', or 'all' (default: all)",
    )
    parser.add_argument(
        "--all",
        dest="all_horizons",
        action="store_true",
        default=False,
        help="Benchmark all horizons (alias for --horizon all)",
    )
    parsed = parser.parse_args()
    if parsed.all_horizons:
        parsed.horizon = "all"
    return parsed


def main() -> int:
    """CLI execution entrypoint."""
    args = parse_args()

    selected_horizons = ["booking", "dispatch"] if args.horizon == "all" else [args.horizon]

    try:
        run_baseline_benchmark(
            config_path=args.config,
            selected_horizons=selected_horizons,
        )
        return 0
    except Exception as e:
        logger.exception("Benchmark failed with error: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
