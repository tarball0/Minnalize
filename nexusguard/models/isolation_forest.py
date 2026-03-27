"""
Isolation Forest Anomaly Detector
Tree-based unsupervised anomaly detection — isolates anomalies by random partitioning.
"""

import numpy as np
from sklearn.ensemble import IsolationForest

from nexusguard.config import ModelConfig


class IsolationForestDetector:
    """Isolation Forest wrapper with score normalization."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = IsolationForest(
            n_estimators=config.if_n_estimators,
            contamination=config.if_contamination,
            max_samples=config.if_max_samples,
            random_state=42,
            n_jobs=-1,
        )
        self._fitted = False

    def fit(self, X: np.ndarray) -> "IsolationForestDetector":
        """Fit the isolation forest on training data."""
        self.model.fit(X)
        self._fitted = True
        return self

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """Return normalized anomaly scores in [0, 1]. Higher = more anomalous."""
        raw_scores = self.model.decision_function(X)
        # decision_function returns negative for anomalies, positive for normal
        # Invert and normalize to [0, 1]
        normalized = 1 - (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-8)
        return normalized

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary predictions (1 = anomaly, 0 = normal)."""
        preds = self.model.predict(X)
        # scikit-learn returns -1 for anomalies, 1 for normal
        return (preds == -1).astype(int)
