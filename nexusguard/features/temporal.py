"""
Temporal Feature Extraction
Rolling-window and time-series features for activity patterns.
"""

import numpy as np
import pandas as pd

from nexusguard.config import FeatureConfig


class TemporalFeatureExtractor:
    """Extracts time-based behavioral features using rolling windows."""

    def __init__(self, config: FeatureConfig):
        self.windows = config.rolling_windows

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features to the dataframe."""
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Per-user rolling statistics using integer-based windows for robustness
        for window in self.windows:
            suffix = f"_{window}m"
            # Use integer window (number of events) as a proxy - simpler and robust
            w = max(window // 5, 2)  # Scale down: ~1 event per 5 min on average
            df[f"event_count{suffix}"] = (
                df.groupby("user_id")["timestamp"]
                .transform(lambda s: s.rolling(w, min_periods=1).count())
                .fillna(0)
            )
            df[f"bytes_sent_mean{suffix}"] = (
                df.groupby("user_id")["bytes_sent"]
                .transform(lambda s: s.rolling(w, min_periods=1).mean())
                .fillna(0)
            )
            df[f"unique_dests{suffix}"] = (
                df.groupby("user_id")["dest_ip"]
                .transform(lambda s: pd.Series(
                    [len(set(s.iloc[max(0, i - w + 1):i + 1])) for i in range(len(s))],
                    index=s.index,
                ))
                .fillna(1)
            )

        # Time since last event per user
        df["time_since_last_event_s"] = (
            df.groupby("user_id")["timestamp"]
            .diff()
            .dt.total_seconds()
            .fillna(0)
        )

        # Burst detection: count events in last 5 events per user
        df["burst_count"] = (
            df.groupby("user_id")["timestamp"]
            .transform(lambda s: s.rolling(5, min_periods=1).count())
            .fillna(0)
        )

        # Failed event rate in last N events per user
        success_col = "success_int" if "success_int" in df.columns else "success"
        df["recent_failure_rate"] = (
            df.groupby("user_id")[success_col]
            .transform(lambda s: s.astype(float).rolling(10, min_periods=1).apply(
                lambda x: 1 - x.mean(), raw=True
            ))
            .fillna(0)
        )

        return df

    def get_feature_names(self) -> list[str]:
        """Return names of generated temporal features."""
        names = []
        for w in self.windows:
            names.extend([
                f"event_count_{w}m",
                f"bytes_sent_mean_{w}m",
                f"unique_dests_{w}m",
            ])
        names.extend([
            "time_since_last_event_s",
            "burst_count",
            "recent_failure_rate",
        ])
        return names
