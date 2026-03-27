"""
Behavioral Profiling
Builds statistical profiles per user/entity and scores deviations.
"""

import numpy as np
import pandas as pd


class BehavioralProfiler:
    """Builds and maintains per-user behavioral profiles for deviation scoring."""

    def __init__(self):
        self.profiles: dict[str, dict] = {}
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "BehavioralProfiler":
        """Build behavioral profiles from historical data."""
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["hour"] = df["timestamp"].dt.hour

        for user_id, group in df.groupby("user_id"):
            self.profiles[user_id] = {
                # Activity timing profile
                "hour_distribution": group["hour"].value_counts(normalize=True).to_dict(),
                "typical_hours": set(group["hour"].mode().tolist()),
                "mean_events_per_day": len(group) / max(group["timestamp"].dt.date.nunique(), 1),
                # Data transfer profile
                "bytes_sent_mean": group["bytes_sent"].mean(),
                "bytes_sent_std": max(group["bytes_sent"].std(), 1),
                "bytes_received_mean": group["bytes_received"].mean(),
                "bytes_received_std": max(group["bytes_received"].std(), 1),
                # Event type profile
                "event_type_dist": group["event_type"].value_counts(normalize=True).to_dict(),
                "common_event_types": set(group["event_type"].value_counts().head(5).index),
                # Destination profile
                "common_destinations": set(group["dest_ip"].value_counts().head(10).index),
                "unique_dest_count": group["dest_ip"].nunique(),
                # Process profile
                "common_processes": set(group["process_name"].value_counts().head(5).index),
                # Success rate
                "success_rate": group["success"].mean(),
            }

        self._fitted = True
        return self

    def score(self, df: pd.DataFrame) -> pd.Series:
        """Score each event based on deviation from the user's behavioral profile."""
        if not self._fitted:
            raise RuntimeError("Profiler not fitted.")

        scores = np.zeros(len(df))
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["hour"] = df["timestamp"].dt.hour

        for idx, row in df.iterrows():
            user = row["user_id"]
            if user not in self.profiles:
                scores[idx] = 0.8  # Unknown user is suspicious
                continue

            profile = self.profiles[user]
            anomaly_signals = []

            # 1. Unusual hour of activity
            hour_prob = profile["hour_distribution"].get(row["hour"], 0)
            anomaly_signals.append(1.0 - min(hour_prob * 10, 1.0))

            # 2. Unusual data transfer volume (z-score)
            z_sent = abs(row["bytes_sent"] - profile["bytes_sent_mean"]) / profile["bytes_sent_std"]
            anomaly_signals.append(min(z_sent / 5.0, 1.0))

            # 3. Unusual event type
            if row["event_type"] not in profile["common_event_types"]:
                anomaly_signals.append(0.6)
            else:
                anomaly_signals.append(0.0)

            # 4. Unusual destination
            if row["dest_ip"] not in profile["common_destinations"]:
                anomaly_signals.append(0.4)
            else:
                anomaly_signals.append(0.0)

            # 5. Unusual process
            if row["process_name"] not in profile["common_processes"]:
                anomaly_signals.append(0.5)
            else:
                anomaly_signals.append(0.0)

            # 6. Failed event when user normally succeeds
            if not row["success"] and profile["success_rate"] > 0.9:
                anomaly_signals.append(0.5)
            else:
                anomaly_signals.append(0.0)

            # Weighted average of signals
            weights = [0.25, 0.25, 0.15, 0.15, 0.10, 0.10]
            scores[idx] = sum(s * w for s, w in zip(anomaly_signals, weights))

        return pd.Series(scores, index=df.index, name="behavioral_score")
