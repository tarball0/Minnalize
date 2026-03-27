"""
Alert Generation Engine
Produces structured alerts from anomaly scores with severity classification.
"""

import json
from datetime import datetime
from typing import Any

import pandas as pd

from nexusguard.config import AlertConfig


class Alert:
    """Represents a single security alert."""

    def __init__(
        self,
        alert_id: str,
        timestamp: datetime,
        user_id: str,
        event_type: str,
        risk_level: str,
        ensemble_score: float,
        model_scores: dict[str, float],
        description: str,
        source_host: str = "",
        dest_ip: str = "",
    ):
        self.alert_id = alert_id
        self.timestamp = timestamp
        self.user_id = user_id
        self.event_type = event_type
        self.risk_level = risk_level
        self.ensemble_score = ensemble_score
        self.model_scores = model_scores
        self.description = description
        self.source_host = source_host
        self.dest_ip = dest_ip

    def to_dict(self) -> dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "timestamp": self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else str(self.timestamp),
            "user_id": self.user_id,
            "event_type": self.event_type,
            "risk_level": self.risk_level,
            "ensemble_score": round(self.ensemble_score, 4),
            "model_scores": {k: round(v, 4) for k, v in self.model_scores.items()},
            "description": self.description,
            "source_host": self.source_host,
            "dest_ip": self.dest_ip,
        }

    def __repr__(self) -> str:
        return f"Alert({self.risk_level}: {self.user_id} - {self.event_type} [{self.ensemble_score:.2f}])"


class AlertEngine:
    """Generates and manages security alerts from anomaly detection results."""

    def __init__(self, config: AlertConfig):
        self.config = config
        self.alerts: list[Alert] = []
        self._alert_counter = 0

    def generate_alerts(self, events_df: pd.DataFrame, scores_df: pd.DataFrame) -> list[Alert]:
        """Generate alerts from scored events."""
        new_alerts = []
        anomalous = scores_df[scores_df["is_anomaly"] == 1]

        for idx in anomalous.index:
            if idx >= len(events_df):
                continue

            event = events_df.iloc[idx]
            score_row = scores_df.iloc[idx]

            self._alert_counter += 1
            alert_id = f"NG-{self._alert_counter:06d}"

            description = self._generate_description(event, score_row)

            alert = Alert(
                alert_id=alert_id,
                timestamp=event.get("timestamp", datetime.now()),
                user_id=event.get("user_id", "unknown"),
                event_type=event.get("event_type", "unknown"),
                risk_level=str(score_row.get("risk_level", "MEDIUM")),
                ensemble_score=float(score_row["ensemble_score"]),
                model_scores={
                    "autoencoder": float(score_row.get("autoencoder_score", 0)),
                    "lstm": float(score_row.get("lstm_score", 0)),
                    "isolation_forest": float(score_row.get("isolation_forest_score", 0)),
                    "behavioral": float(score_row.get("behavioral_score", 0)),
                },
                description=description,
                source_host=event.get("source_host", ""),
                dest_ip=event.get("dest_ip", ""),
            )
            new_alerts.append(alert)

        self.alerts.extend(new_alerts)
        return new_alerts

    def _generate_description(self, event: pd.Series, scores: pd.Series) -> str:
        """Generate a human-readable alert description."""
        parts = []

        # Identify primary contributing models
        model_scores = {
            "Autoencoder (reconstruction)": scores.get("autoencoder_score", 0),
            "LSTM (sequence)": scores.get("lstm_score", 0),
            "Isolation Forest (statistical)": scores.get("isolation_forest_score", 0),
            "Behavioral Profile": scores.get("behavioral_score", 0),
        }
        top_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)[:2]
        flagged_by = [m for m, s in top_models if s > 0.3]

        event_type = event.get("event_type", "unknown")
        user = event.get("user_id", "unknown")

        parts.append(f"Anomalous {event_type} detected for {user}.")

        if flagged_by:
            parts.append(f"Flagged by: {', '.join(flagged_by)}.")

        agreement = int(scores.get("agreement_count", 0))
        if agreement >= 3:
            parts.append(f"High model agreement ({agreement}/4 detectors triggered).")

        bytes_sent = event.get("bytes_sent", 0)
        if bytes_sent > 100_000:
            parts.append(f"Large data transfer: {bytes_sent:,} bytes.")

        return " ".join(parts)

    def get_summary(self) -> dict:
        """Return alert summary statistics."""
        if not self.alerts:
            return {"total": 0}

        df = pd.DataFrame([a.to_dict() for a in self.alerts])
        return {
            "total": len(self.alerts),
            "by_risk_level": df["risk_level"].value_counts().to_dict(),
            "by_event_type": df["event_type"].value_counts().to_dict(),
            "top_users": df["user_id"].value_counts().head(10).to_dict(),
            "avg_score": float(df["ensemble_score"].mean()),
        }

    def export_json(self, filepath: str) -> None:
        """Export all alerts as JSON."""
        data = [a.to_dict() for a in self.alerts]
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
