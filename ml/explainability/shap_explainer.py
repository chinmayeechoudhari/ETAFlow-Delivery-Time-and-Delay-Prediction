"""
ETAFlow SHAP Model Explainability Module.

Computes TreeExplainer Shapley values for global feature rankings and generates
structured, machine-readable local explanations with human-readable drivers
for operations.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import shap  # pyrefly: ignore [missing-import]

logger = logging.getLogger(__name__)


class ETAShapExplainer:
    """Production SHAP explainer for tree-based regression and classification models."""

    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        target_type: str = "regression",
    ) -> None:
        """Initialize TreeExplainer with model and feature names.

        Args:
            model: Trained LightGBM or XGBoost model.
            feature_names: List of preprocessed feature column names.
            target_type: "regression" or "classification".
        """
        self.model = model
        self.feature_names = list(feature_names)
        self.target_type = target_type.lower()
        logger.info("Initializing SHAP TreeExplainer for %s model...", self.target_type)
        self.explainer = shap.TreeExplainer(model)

    def explain_global(self, X: pd.DataFrame, top_n: int = 20) -> Dict[str, Any]:
        """Compute global feature importance based on mean absolute SHAP values.

        Args:
            X: Evaluation feature matrix (or sample subset).
            top_n: Number of top features to return in detailed ranking.

        Returns:
            Dict[str, Any]: Top features, mean absolute SHAP values, and rankings.
        """
        X_df = pd.DataFrame(X, columns=self.feature_names)
        shap_values = self.explainer.shap_values(X_df)

        # Handle classification where shap_values might be a list [class_0, class_1]
        if isinstance(shap_values, list):
            sv = shap_values[1]  # positive class (delay)
        elif len(np.shape(shap_values)) == 3:
            sv = shap_values[:, :, 1]
        else:
            sv = shap_values

        mean_abs_shap = np.mean(np.abs(sv), axis=0)
        ranked_indices = np.argsort(mean_abs_shap)[::-1]

        rankings: List[Dict[str, Any]] = []
        for rank, idx in enumerate(ranked_indices[:top_n], start=1):
            rankings.append({
                "rank": rank,
                "feature": self.feature_names[idx],
                "mean_abs_shap": round(float(mean_abs_shap[idx]), 5),
            })

        return {
            "target_type": self.target_type,
            "total_features": len(self.feature_names),
            "top_features": rankings,
            "summary": {r["feature"]: r["mean_abs_shap"] for r in rankings},
        }

    def explain_instance(
        self,
        x_row: Union[pd.Series, pd.DataFrame],
        base_value: Optional[float] = None,
        prediction: Optional[float] = None,
        top_k: int = 5,
        shipment_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate local explanation for a single shipment with driver narratives.

        Args:
            x_row: Single row feature vector.
            base_value: Optional model baseline expected value.
            prediction: Optional predicted model output.
            top_k: Number of positive and negative drivers to highlight.
            shipment_id: Optional tracking identifier.

        Returns:
            Dict[str, Any]: Structured machine-readable explanation and narrative text.
        """
        if isinstance(x_row, pd.Series):
            X_sample = pd.DataFrame([x_row], columns=self.feature_names)
        else:
            X_sample = pd.DataFrame(x_row, columns=self.feature_names)

        shap_values = self.explainer.shap_values(X_sample)
        expected_val = self.explainer.expected_value

        if isinstance(shap_values, list):
            sv = shap_values[1][0]
            base_v = (
                float(expected_val[1])
                if isinstance(expected_val, (list, np.ndarray))
                else float(expected_val)
            )
        elif len(np.shape(shap_values)) == 3:
            sv = shap_values[0, :, 1]
            base_v = (
                float(expected_val[1])
                if isinstance(expected_val, (list, np.ndarray))
                else float(expected_val)
            )
        else:
            sv = shap_values[0]
            base_v = (
                float(expected_val[0])
                if isinstance(expected_val, (list, np.ndarray))
                else float(expected_val)
            )

        feature_contribs = []
        for idx, feat_name in enumerate(self.feature_names):
            val = float(X_sample.iloc[0, idx])
            shap_val = float(sv[idx])
            direction = (
                "increases_eta_or_risk"
                if shap_val > 0
                else "decreases_eta_or_risk"
            )
            feature_contribs.append({
                "feature": feat_name,
                "feature_value": round(val, 4),
                "shap_value": round(shap_val, 4),
                "direction": direction,
            })

        # Separate positive and negative contributors
        pos_drivers = sorted(
            [c for c in feature_contribs if c["shap_value"] > 0],
            key=lambda x: x["shap_value"],
            reverse=True,
        )[:top_k]
        neg_drivers = sorted(
            [c for c in feature_contribs if c["shap_value"] < 0],
            key=lambda x: x["shap_value"],
        )[:top_k]

        pred_val = (
            prediction
            if prediction is not None
            else float(base_v + np.sum(sv))
        )

        # Synthesize dynamic human-readable explanation narrative
        label_target = (
            "ETA (days)"
            if self.target_type == "regression"
            else "Delay Probability"
        )

        narrative_lines = [
            f"Shipment: {shipment_id or 'UNKNOWN'}",
            f"Predicted {label_target}: {pred_val:.2f} (Baseline: {base_v:.2f})",
        ]

        if self.target_type == "regression":
            narrative_lines.append("ETA increased because:")
            for d in pos_drivers:
                narrative_lines.append(
                    f"  - {d['feature']} = {d['feature_value']} "
                    f"(+{d['shap_value']:.2f} days)"
                )
            narrative_lines.append("ETA decreased because:")
            for d in neg_drivers:
                narrative_lines.append(
                    f"  - {d['feature']} = {d['feature_value']} "
                    f"({d['shap_value']:.2f} days)"
                )
        else:
            narrative_lines.append("Delay risk increased because:")
            for d in pos_drivers:
                narrative_lines.append(
                    f"  - {d['feature']} = {d['feature_value']} "
                    f"(+{d['shap_value']:.3f} log-odds)"
                )
            narrative_lines.append("Delay risk reduced because:")
            for d in neg_drivers:
                narrative_lines.append(
                    f"  - {d['feature']} = {d['feature_value']} "
                    f"({d['shap_value']:.3f} log-odds)"
                )

        explanation_narrative = "\n".join(narrative_lines)

        return {
            "shipment_id": shipment_id,
            "target_type": self.target_type,
            "baseline_value": round(base_v, 4),
            "predicted_value": round(pred_val, 4),
            "positive_drivers": pos_drivers,
            "negative_drivers": neg_drivers,
            "narrative": explanation_narrative,
        }
