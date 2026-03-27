"""
analysis.py
-----------
Three-layer anomaly detection engine.

Layer 1 – Rule-based heuristics (deterministic, fast)
    Hard thresholds based on security domain knowledge.
    Always run; results labelled with MEDIUM/HIGH/CRITICAL severity.

Layer 2 – Z-score statistical baseline
    Flags features that deviate significantly from the population mean.
    Adapts automatically as the baseline grows.

Layer 3 – Isolation Forest (ML)
    Unsupervised anomaly detection across the full feature vector.
    Learns the shape of "normal" and scores outliers.
    Contamination is set conservatively so only genuine outliers surface.

All three layers produce a list of Finding objects that are merged and
deduplicated before being handed to the alerting module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ── Tuneable thresholds ────────────────────────────────────────────────────────

BRUTE_FORCE_FAILURE_THRESHOLD  = 10    # ≥ N failures from one IP  → CRITICAL
HIGH_FAILURE_RATE_THRESHOLD    = 0.75  # ≥ 75 % failures            → HIGH
AFTER_HOURS_THRESHOLD          = 2     # ≥ N after-hours events     → MEDIUM
UNIQUE_USERS_PER_IP_THRESHOLD  = 4     # one IP targeting many users → HIGH
UNIQUE_IPS_PER_USER_THRESHOLD  = 5     # one user from many IPs     → HIGH
ZSCORE_THRESHOLD               = 2.5   # standard deviations        → MEDIUM
IF_CONTAMINATION               = 0.05  # assumed 5 % anomaly rate


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class Finding:
    entity_type:  str            # "ip" | "user"
    entity_value: str            # the IP or username
    severity:     str            # LOW | MEDIUM | HIGH | CRITICAL
    category:     str            # brief label, e.g. "brute_force"
    description:  str            # human-readable explanation
    layer:        str            # "rules" | "zscore" | "isolation_forest"
    evidence:     dict = field(default_factory=dict)  # raw numbers that triggered this

    def __eq__(self, other):
        return (self.entity_type, self.entity_value, self.category) == \
               (other.entity_type, other.entity_value, other.category)

    def __hash__(self):
        return hash((self.entity_type, self.entity_value, self.category))


# ── Layer 1: Rule-based ────────────────────────────────────────────────────────

def _rules_ip(ip_feats: pd.DataFrame) -> list[Finding]:
    findings = []
    for _, row in ip_feats.iterrows():
        ip = row["source_ip"]

        if row["failure_count"] >= BRUTE_FORCE_FAILURE_THRESHOLD:
            findings.append(Finding(
                entity_type="ip", entity_value=ip,
                severity="CRITICAL", category="brute_force",
                description=(
                    f"IP {ip} generated {int(row['failure_count'])} login failures "
                    f"(threshold: {BRUTE_FORCE_FAILURE_THRESHOLD})."
                ),
                layer="rules",
                evidence={"failure_count": int(row["failure_count"]),
                          "failure_rate": round(row["failure_rate"], 3)},
            ))

        if row["failure_rate"] >= HIGH_FAILURE_RATE_THRESHOLD and row["total_events"] >= 5:
            findings.append(Finding(
                entity_type="ip", entity_value=ip,
                severity="HIGH", category="high_failure_rate",
                description=(
                    f"IP {ip} has a {row['failure_rate']:.0%} login failure rate "
                    f"over {int(row['total_events'])} events."
                ),
                layer="rules",
                evidence={"failure_rate": round(row["failure_rate"], 3),
                          "total_events": int(row["total_events"])},
            ))

        if row["unique_users"] >= UNIQUE_USERS_PER_IP_THRESHOLD:
            findings.append(Finding(
                entity_type="ip", entity_value=ip,
                severity="HIGH", category="credential_stuffing",
                description=(
                    f"IP {ip} targeted {int(row['unique_users'])} different usernames "
                    f"— possible credential stuffing."
                ),
                layer="rules",
                evidence={"unique_users": int(row["unique_users"])},
            ))

        if row["after_hours_count"] >= AFTER_HOURS_THRESHOLD:
            findings.append(Finding(
                entity_type="ip", entity_value=ip,
                severity="MEDIUM", category="after_hours_activity",
                description=(
                    f"IP {ip} had {int(row['after_hours_count'])} events outside business hours."
                ),
                layer="rules",
                evidence={"after_hours_count": int(row["after_hours_count"])},
            ))

    return findings


def _rules_user(user_feats: pd.DataFrame) -> list[Finding]:
    findings = []
    for _, row in user_feats.iterrows():
        user = row["username"]

        if row["unique_ips"] >= UNIQUE_IPS_PER_USER_THRESHOLD:
            findings.append(Finding(
                entity_type="user", entity_value=user,
                severity="HIGH", category="impossible_travel",
                description=(
                    f"User '{user}' authenticated from {int(row['unique_ips'])} distinct IPs "
                    f"— possible account sharing or impossible travel."
                ),
                layer="rules",
                evidence={"unique_ips": int(row["unique_ips"])},
            ))

        if row["after_hours_count"] >= AFTER_HOURS_THRESHOLD:
            findings.append(Finding(
                entity_type="user", entity_value=user,
                severity="MEDIUM", category="after_hours_user",
                description=(
                    f"User '{user}' was active {int(row['after_hours_count'])} times "
                    f"outside business hours."
                ),
                layer="rules",
                evidence={"after_hours_count": int(row["after_hours_count"])},
            ))

    return findings


# ── Layer 2: Z-score baseline ──────────────────────────────────────────────────

_NUMERIC_IP_COLS   = ["total_events", "failure_count", "failure_rate",
                       "unique_users", "after_hours_count", "event_rate_per_min"]
_NUMERIC_USER_COLS = ["total_events", "failure_count", "failure_rate",
                       "unique_ips", "sudo_count", "after_hours_count"]


def _zscore_findings(feats: pd.DataFrame, id_col: str,
                     numeric_cols: list[str], entity_type: str) -> list[Finding]:
    findings = []
    cols = [c for c in numeric_cols if c in feats.columns]
    data = feats[cols].fillna(0)

    means = data.mean()
    stds  = data.std(ddof=0).replace(0, np.nan)   # avoid /0 for constant columns
    zscores = ((data - means) / stds).fillna(0)

    for idx, row in zscores.iterrows():
        flagged = row[row.abs() > ZSCORE_THRESHOLD]
        if flagged.empty:
            continue
        entity = feats.loc[idx, id_col]
        for col, z in flagged.items():
            raw_val = feats.loc[idx, col]
            findings.append(Finding(
                entity_type=entity_type, entity_value=str(entity),
                severity="MEDIUM", category=f"zscore_{col}",
                description=(
                    f"{'IP' if entity_type == 'ip' else 'User'} '{entity}' has an unusually "
                    f"high '{col}' value ({raw_val:.2f}), Z-score={z:.2f}."
                ),
                layer="zscore",
                evidence={"column": col, "value": round(float(raw_val), 4),
                          "zscore": round(float(z), 4)},
            ))
    return findings


# ── Layer 3: Isolation Forest ──────────────────────────────────────────────────

def _isolation_forest_findings(feats: pd.DataFrame, id_col: str,
                                numeric_cols: list[str], entity_type: str) -> list[Finding]:
    """Score every entity; return those the model marks as anomalies (-1)."""
    cols = [c for c in numeric_cols if c in feats.columns]
    if len(feats) < 5:
        # Not enough samples for meaningful ML training
        return []

    X = feats[cols].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=IF_CONTAMINATION,
        random_state=42,
        n_jobs=-1,
    )
    predictions = model.fit_predict(X_scaled)   # -1 = anomaly, 1 = normal
    scores      = model.score_samples(X_scaled)  # lower = more anomalous

    findings = []
    for i, (pred, score) in enumerate(zip(predictions, scores)):
        if pred == -1:
            entity = feats.iloc[i][id_col]
            findings.append(Finding(
                entity_type=entity_type, entity_value=str(entity),
                severity="HIGH", category="ml_outlier",
                description=(
                    f"{'IP' if entity_type == 'ip' else 'User'} '{entity}' was flagged as "
                    f"an outlier by Isolation Forest (anomaly score: {score:.4f})."
                ),
                layer="isolation_forest",
                evidence={"anomaly_score": round(float(score), 4)},
            ))
    return findings


# ── Public API ─────────────────────────────────────────────────────────────────

SEVERITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}


def run_analysis(ip_feats: pd.DataFrame,
                 user_feats: pd.DataFrame) -> list[Finding]:
    """
    Run all three detection layers and return a deduplicated, sorted list
    of Finding objects.

    Deduplication key: (entity_type, entity_value, category) – if the same
    entity triggers both a rule and the ML model for the same category, the
    higher-severity finding is kept.
    """
    all_findings: list[Finding] = []

    # Layer 1
    all_findings += _rules_ip(ip_feats)
    all_findings += _rules_user(user_feats)

    # Layer 2
    all_findings += _zscore_findings(ip_feats,   "source_ip", _NUMERIC_IP_COLS,   "ip")
    all_findings += _zscore_findings(user_feats, "username",  _NUMERIC_USER_COLS, "user")

    # Layer 3
    all_findings += _isolation_forest_findings(ip_feats,   "source_ip", _NUMERIC_IP_COLS,   "ip")
    all_findings += _isolation_forest_findings(user_feats, "username",  _NUMERIC_USER_COLS, "user")

    # Deduplicate: keep highest-severity per (entity, category)
    seen: dict[tuple, Finding] = {}
    for f in all_findings:
        key = (f.entity_type, f.entity_value, f.category)
        if key not in seen or SEVERITY_ORDER[f.severity] < SEVERITY_ORDER[seen[key].severity]:
            seen[key] = f

    # Sort: severity first, then entity type, then entity value
    return sorted(
        seen.values(),
        key=lambda f: (SEVERITY_ORDER[f.severity], f.entity_type, f.entity_value),
    )
