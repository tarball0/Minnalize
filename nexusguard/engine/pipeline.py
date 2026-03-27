"""
Main Processing Pipeline
Orchestrates data flow through feature extraction, multiple models, and ensemble scoring.
"""

import time
import logging

import numpy as np
import pandas as pd

from nexusguard.config import NexusGuardConfig
from nexusguard.data.preprocessor import Preprocessor
from nexusguard.features.temporal import TemporalFeatureExtractor
from nexusguard.features.behavioral import BehavioralProfiler
from nexusguard.features.graph_features import GraphFeatureExtractor
from nexusguard.models.autoencoder import AutoencoderDetector
from nexusguard.models.lstm_detector import LSTMDetector
from nexusguard.models.isolation_forest import IsolationForestDetector
from nexusguard.models.ensemble import EnsembleScorer
from nexusguard.engine.alerter import AlertEngine

logger = logging.getLogger("nexusguard")


class NexusGuardPipeline:
    """End-to-end anomaly detection pipeline."""

    def __init__(self, config: NexusGuardConfig):
        self.config = config
        self.preprocessor = Preprocessor()
        self.temporal_extractor = TemporalFeatureExtractor(config.features)
        self.behavioral_profiler = BehavioralProfiler()
        self.graph_extractor = GraphFeatureExtractor(config.features)
        self.ensemble = EnsembleScorer(config.model)
        self.alert_engine = AlertEngine(config.alert)

        # Models initialized after feature dimension is known
        self.autoencoder: AutoencoderDetector | None = None
        self.lstm: LSTMDetector | None = None
        self.isolation_forest: IsolationForestDetector | None = None

        self._trained = False
        self.training_metrics: dict = {}
        self.feature_columns: list[str] = []

    def train(self, df: pd.DataFrame) -> dict:
        """Train all models on historical data."""
        logger.info("Starting NexusGuard training pipeline...")
        t0 = time.time()
        metrics = {}

        # Step 1: Preprocess
        logger.info("Step 1/6: Preprocessing data...")
        processed = self.preprocessor.fit_transform(df)

        # Step 2: Extract temporal features
        logger.info("Step 2/6: Extracting temporal features...")
        processed = self.temporal_extractor.extract(processed)

        # Step 3: Build behavioral profiles
        logger.info("Step 3/6: Building behavioral profiles...")
        self.behavioral_profiler.fit(df)

        # Step 4: Build graph baseline
        logger.info("Step 4/6: Building graph baseline...")
        normal_data = df[df["is_anomaly"] == 0] if "is_anomaly" in df.columns else df
        self.graph_extractor.build_baseline(normal_data)
        processed = self.graph_extractor.extract(processed)

        # Determine feature columns
        self.feature_columns = self.preprocessor.get_feature_columns()
        temporal_features = [c for c in processed.columns
                            if any(c.startswith(p) for p in
                                   ["event_count_", "bytes_sent_mean_", "unique_dests_",
                                    "time_since_last_event", "burst_count", "recent_failure"])]
        graph_features = ["graph_anomaly_score", "is_new_edge", "is_cross_community"]
        self.feature_columns += temporal_features + graph_features
        self.feature_columns = [c for c in self.feature_columns if c in processed.columns]

        X = processed[self.feature_columns].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        input_dim = X.shape[1]
        logger.info(f"   Feature dimension: {input_dim}")

        # Step 5: Train models
        logger.info("Step 5/6: Training Autoencoder...")
        self.autoencoder = AutoencoderDetector(self.config.model, input_dim, self.config.device)
        ae_history = self.autoencoder.fit(X)
        metrics["autoencoder_final_loss"] = ae_history["loss"][-1] if ae_history["loss"] else 0

        logger.info("Step 5/6: Training LSTM Sequence Detector...")
        self.lstm = LSTMDetector(self.config.model, input_dim, self.config.device)
        lstm_history = self.lstm.fit(X)
        metrics["lstm_final_loss"] = lstm_history["loss"][-1] if lstm_history["loss"] else 0

        logger.info("Step 5/6: Training Isolation Forest...")
        self.isolation_forest = IsolationForestDetector(self.config.model)
        self.isolation_forest.fit(X)

        # Step 6: Compute training scores for calibration
        logger.info("Step 6/6: Calibrating ensemble...")
        ae_scores = self.autoencoder.predict_scores(X)
        lstm_scores = self.lstm.predict_scores(X)
        if_scores = self.isolation_forest.predict_scores(X)
        beh_scores = self.behavioral_profiler.score(df).values

        metrics["training_time_seconds"] = round(time.time() - t0, 2)
        metrics["num_features"] = input_dim
        metrics["num_samples"] = len(X)
        self.training_metrics = metrics
        self._trained = True

        logger.info(f"Training complete in {metrics['training_time_seconds']}s")
        return metrics

    def predict(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Run anomaly detection on new data. Returns (events_df, scores_df)."""
        if not self._trained:
            raise RuntimeError("Pipeline not trained. Call train() first.")

        # Preprocess
        processed = self.preprocessor.transform(df)
        processed = self.temporal_extractor.extract(processed)
        processed = self.graph_extractor.extract(processed)

        X = processed[self.feature_columns].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Get scores from each model
        ae_scores = self.autoencoder.predict_scores(X)
        lstm_scores = self.lstm.predict_scores(X)
        if_scores = self.isolation_forest.predict_scores(X)
        beh_scores = self.behavioral_profiler.score(df).values

        # Ensemble
        scores_df = self.ensemble.score(ae_scores, lstm_scores, if_scores, beh_scores)
        return df, scores_df

    def detect_and_alert(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list]:
        """Full detection pipeline with alert generation."""
        events_df, scores_df = self.predict(df)
        alerts = self.alert_engine.generate_alerts(events_df, scores_df)
        return events_df, scores_df, alerts

    def evaluate(self, df: pd.DataFrame) -> dict:
        """Evaluate the pipeline on labeled data."""
        if "is_anomaly" not in df.columns:
            raise ValueError("Evaluation requires 'is_anomaly' column.")

        _, scores_df = self.predict(df)
        true_labels = df["is_anomaly"].values
        pred_labels = scores_df["is_anomaly"].values

        tp = ((pred_labels == 1) & (true_labels == 1)).sum()
        fp = ((pred_labels == 1) & (true_labels == 0)).sum()
        fn = ((pred_labels == 0) & (true_labels == 1)).sum()
        tn = ((pred_labels == 0) & (true_labels == 0)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0

        return {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_negatives": int(tn),
            "total_anomalies_detected": int(pred_labels.sum()),
            "total_actual_anomalies": int(true_labels.sum()),
        }
