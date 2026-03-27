"""
Data Preprocessor
Cleans, encodes, and normalizes raw activity logs for model consumption.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


class Preprocessor:
    """Cleans and encodes raw activity log data."""

    def __init__(self):
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self.categorical_cols = [
            "user_id", "event_type", "source_host", "protocol",
            "severity", "process_name",
        ]
        self.numerical_cols = [
            "port", "bytes_sent", "bytes_received", "duration_ms",
            "success_int", "hour", "day_of_week", "is_weekend",
        ]
        self._fitted = False

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit encoders/scalers and transform the data."""
        df = self._add_time_features(df.copy())
        df = self._encode_categoricals(df, fit=True)
        df = self._scale_numericals(df, fit=True)
        self._fitted = True
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using already-fitted encoders/scalers."""
        if not self._fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit_transform first.")
        df = self._add_time_features(df.copy())
        df = self._encode_categoricals(df, fit=False)
        df = self._scale_numericals(df, fit=False)
        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract temporal features from timestamp."""
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["success_int"] = df["success"].astype(int)
        df["minutes_since_midnight"] = df["hour"] * 60 + df["timestamp"].dt.minute
        # Log-transform skewed numerical features
        for col in ["bytes_sent", "bytes_received", "duration_ms"]:
            df[f"{col}_log"] = np.log1p(df[col].clip(lower=0))
        return df

    def _encode_categoricals(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Label encode categorical columns."""
        for col in self.categorical_cols:
            if col not in df.columns:
                continue
            if fit:
                le = LabelEncoder()
                df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders[col]
                # Handle unseen labels
                known = set(le.classes_)
                df[f"{col}_enc"] = df[col].astype(str).map(
                    lambda x, _known=known, _le=le: (
                        _le.transform([x])[0] if x in _known else -1
                    )
                )
        return df

    def _scale_numericals(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Standardize numerical columns."""
        cols_to_scale = [c for c in self.numerical_cols if c in df.columns]
        log_cols = ["bytes_sent_log", "bytes_received_log", "duration_ms_log"]
        cols_to_scale += [c for c in log_cols if c in df.columns]

        if fit:
            df[cols_to_scale] = self.scaler.fit_transform(df[cols_to_scale])
        else:
            df[cols_to_scale] = self.scaler.transform(df[cols_to_scale])
        return df

    def get_feature_columns(self) -> list[str]:
        """Return the list of feature columns used for modeling."""
        encoded_cats = [f"{c}_enc" for c in self.categorical_cols]
        log_cols = ["bytes_sent_log", "bytes_received_log", "duration_ms_log"]
        return self.numerical_cols + encoded_cats + log_cols + ["minutes_since_midnight"]
