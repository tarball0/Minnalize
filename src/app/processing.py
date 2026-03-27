"""
processing.py
-------------
Derives per-IP and per-user behavioural features from the raw event log.

Features computed
-----------------
Per-IP aggregates (ip_features):
  total_events        – total events from this IP
  failure_count       – number of login_failure events
  success_count       – number of login_success events
  failure_rate        – failure_count / total_events
  unique_users        – distinct usernames seen from this IP
  sudo_count          – number of sudo events
  after_hours_count   – events outside 08:00–17:59
  event_rate_per_min  – events / observation window in minutes

Per-user aggregates (user_features):
  total_events        – total events by this user
  failure_count       – login failures
  failure_rate        – failure_count / total_events
  unique_ips          – distinct source IPs for this user
  sudo_count          – sudo events
  after_hours_count   – events outside business hours

These DataFrames are used by the analysis engine.
"""

import pandas as pd
import numpy as np

BUSINESS_HOURS = range(8, 18)   # 08:00 – 17:59


def _after_hours(series: pd.Series) -> pd.Series:
    """Return boolean mask: True when event is outside business hours."""
    return ~series.dt.hour.isin(BUSINESS_HOURS)


def compute_ip_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate event-level data into per-source-IP behavioural features."""
    window_minutes = max(
        (df["timestamp"].max() - df["timestamp"].min()).total_seconds() / 60,
        1,   # avoid divide-by-zero for tiny datasets
    )

    grp = df.groupby("source_ip")

    ip_feats = pd.DataFrame({
        "total_events":      grp.size(),
        "failure_count":     grp.apply(lambda g: (g["action"] == "login_failure").sum()),
        "success_count":     grp.apply(lambda g: (g["action"] == "login_success").sum()),
        "unique_users":      grp["username"].nunique(),
        "sudo_count":        grp.apply(lambda g: (g["action"] == "sudo").sum()),
        "after_hours_count": grp.apply(lambda g: _after_hours(g["timestamp"]).sum()),
    })

    ip_feats["failure_rate"]       = ip_feats["failure_count"] / ip_feats["total_events"].clip(lower=1)
    ip_feats["event_rate_per_min"] = ip_feats["total_events"] / window_minutes

    return ip_feats.reset_index()


def compute_user_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate event-level data into per-user behavioural features."""
    grp = df.groupby("username")

    user_feats = pd.DataFrame({
        "total_events":      grp.size(),
        "failure_count":     grp.apply(lambda g: (g["action"] == "login_failure").sum()),
        "unique_ips":        grp["source_ip"].nunique(),
        "sudo_count":        grp.apply(lambda g: (g["action"] == "sudo").sum()),
        "after_hours_count": grp.apply(lambda g: _after_hours(g["timestamp"]).sum()),
    })

    user_feats["failure_rate"] = (
        user_feats["failure_count"] / user_feats["total_events"].clip(lower=1)
    )

    return user_feats.reset_index()


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full pipeline: compute both IP and user feature tables.

    Returns:
        (ip_features, user_features) – two DataFrames ready for analysis.
    """
    ip_feats   = compute_ip_features(df)
    user_feats = compute_user_features(df)
    return ip_feats, user_feats
