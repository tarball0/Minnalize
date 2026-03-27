"""
Ensemble Meta-Scorer
Combines multiple model outputs with adaptive weighting for final anomaly decisions.
"""

import numpy as np
import pandas as pd

from nexusguard.config import ModelConfig


class EnsembleScorer:
    """Combines anomaly scores from multiple detectors into a unified score."""

    def __init__(self, config: ModelConfig):
        self.weights = config.ensemble_weights
        self.threshold = config.anomaly_threshold
        self._score_history: list[dict] = []

    def score(
        self,
        autoencoder_scores: np.ndarray,
        lstm_scores: np.ndarray,
        isolation_forest_scores: np.ndarray,
        behavioral_scores: np.ndarray,
    ) -> pd.DataFrame:
        """Compute weighted ensemble anomaly scores."""
        w = self.weights

        ensemble_score = (
            w["autoencoder"] * autoencoder_scores
            + w["lstm"] * lstm_scores
            + w["isolation_forest"] * isolation_forest_scores
            + w["behavioral"] * behavioral_scores
        )

        # Agreement bonus: if multiple models flag something, boost the score
        model_flags = np.column_stack([
            (autoencoder_scores > 0.5).astype(float),
            (lstm_scores > 0.5).astype(float),
            (isolation_forest_scores > 0.5).astype(float),
            (behavioral_scores > 0.5).astype(float),
        ])
        agreement_count = model_flags.sum(axis=1)
        agreement_bonus = np.where(agreement_count >= 3, 0.1, 0.0)
        agreement_bonus += np.where(agreement_count == 4, 0.1, 0.0)

        final_score = np.clip(ensemble_score + agreement_bonus, 0, 1)

        results = pd.DataFrame({
            "autoencoder_score": autoencoder_scores,
            "lstm_score": lstm_scores,
            "isolation_forest_score": isolation_forest_scores,
            "behavioral_score": behavioral_scores,
            "ensemble_score": final_score,
            "agreement_count": agreement_count.astype(int),
            "is_anomaly": (final_score >= self.threshold).astype(int),
            "risk_level": pd.cut(
                final_score,
                bins=[-0.01, 0.3, 0.5, 0.7, 0.85, 1.01],
                labels=["LOW", "MEDIUM", "HIGH", "CRITICAL", "SEVERE"],
            ),
        })
        return results

    def adapt_weights(self, feedback: pd.DataFrame) -> None:
        """Adapt ensemble weights based on feedback (true labels).
        Uses a simple reward mechanism: models that correctly flagged anomalies
        get weight increases; incorrect flags get decreases.
        """
        if "true_label" not in feedback.columns:
            return

        model_names = ["autoencoder", "lstm", "isolation_forest", "behavioral"]
        score_cols = [f"{m}_score" for m in model_names]

        for model, col in zip(model_names, score_cols):
            if col not in feedback.columns:
                continue
            predictions = (feedback[col] > 0.5).astype(int)
            true_labels = feedback["true_label"].astype(int)

            tp = ((predictions == 1) & (true_labels == 1)).sum()
            fp = ((predictions == 1) & (true_labels == 0)).sum()
            fn = ((predictions == 0) & (true_labels == 1)).sum()

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

            # Adjust weight proportional to F1 score
            self.weights[model] *= (0.5 + f1)

        # Re-normalize weights
        total = sum(self.weights.values())
        for k in self.weights:
            self.weights[k] /= total
