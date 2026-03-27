"""
tests/test_analysis.py
-----------------------
Unit tests for the processing and analysis modules.

Run with:  pytest tests/ -v
"""

import pandas as pd
import pytest
from datetime import datetime, timedelta

from src.app.processing import compute_ip_features, compute_user_features
from src.app.analysis   import (
    _rules_ip, _rules_user, run_analysis,
    BRUTE_FORCE_FAILURE_THRESHOLD, UNIQUE_USERS_PER_IP_THRESHOLD,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _make_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@pytest.fixture
def clean_df():
    """100 normal business-hour login events, spread across 5 users/IPs."""
    base = datetime(2026, 3, 27, 9, 0, 0)
    rows = []
    users = ["alice", "bob", "carol", "dave", "eve"]
    ips   = [f"192.168.1.{10 + i}" for i in range(5)]
    for i in range(100):
        rows.append({
            "timestamp": (base + timedelta(minutes=i * 5)).isoformat(sep=" "),
            "source_ip": ips[i % 5],
            "username":  users[i % 5],
            "action":    "login_success",
        })
    return _make_df(rows)


@pytest.fixture
def brute_force_df():
    """Simulated brute-force: 30 rapid failures from a single external IP."""
    base = datetime(2026, 3, 27, 2, 0, 0)   # 2 AM
    rows = [
        {
            "timestamp": (base + timedelta(seconds=i * 2)).isoformat(sep=" "),
            "source_ip": "10.99.0.13",
            "username":  "alice",
            "action":    "login_failure",
        }
        for i in range(30)
    ]
    return _make_df(rows)


@pytest.fixture
def multi_user_ip_df():
    """One IP targeting many different usernames."""
    base = datetime(2026, 3, 27, 10, 0, 0)
    usernames = [f"user_{i}" for i in range(6)]
    rows = [
        {
            "timestamp": (base + timedelta(minutes=i)).isoformat(sep=" "),
            "source_ip": "192.168.2.99",
            "username":  u,
            "action":    "login_failure",
        }
        for i, u in enumerate(usernames)
    ]
    return _make_df(rows)


# ── Processing tests ───────────────────────────────────────────────────────────

class TestProcessing:
    def test_ip_features_shape(self, clean_df):
        ip_feats = compute_ip_features(clean_df)
        assert "source_ip"    in ip_feats.columns
        assert "failure_count" in ip_feats.columns
        assert "failure_rate"  in ip_feats.columns
        assert len(ip_feats)  == 5   # 5 distinct IPs

    def test_user_features_shape(self, clean_df):
        user_feats = compute_user_features(clean_df)
        assert "username"     in user_feats.columns
        assert "unique_ips"   in user_feats.columns
        assert len(user_feats) == 5  # 5 distinct users

    def test_clean_log_has_zero_failures(self, clean_df):
        ip_feats = compute_ip_features(clean_df)
        assert ip_feats["failure_count"].sum() == 0
        assert ip_feats["failure_rate"].sum()  == 0.0

    def test_after_hours_count_is_zero_for_business_hours(self, clean_df):
        ip_feats = compute_ip_features(clean_df)
        assert ip_feats["after_hours_count"].sum() == 0

    def test_brute_force_failure_count(self, brute_force_df):
        ip_feats = compute_ip_features(brute_force_df)
        row = ip_feats[ip_feats["source_ip"] == "10.99.0.13"].iloc[0]
        assert row["failure_count"] == 30
        assert row["failure_rate"]  == pytest.approx(1.0)

    def test_after_hours_detected(self, brute_force_df):
        """All 30 events are at 2 AM — all should be after-hours."""
        ip_feats = compute_ip_features(brute_force_df)
        row = ip_feats[ip_feats["source_ip"] == "10.99.0.13"].iloc[0]
        assert row["after_hours_count"] == 30


# ── Rules-engine tests ─────────────────────────────────────────────────────────

class TestRules:
    def test_brute_force_triggers_critical(self, brute_force_df):
        ip_feats = compute_ip_features(brute_force_df)
        findings = _rules_ip(ip_feats)
        critical = [f for f in findings if f.severity == "CRITICAL" and f.category == "brute_force"]
        assert len(critical) >= 1
        assert critical[0].entity_value == "10.99.0.13"

    def test_clean_log_no_critical(self, clean_df):
        ip_feats = compute_ip_features(clean_df)
        findings = _rules_ip(ip_feats)
        assert not any(f.severity == "CRITICAL" for f in findings)

    def test_credential_stuffing_detected(self, multi_user_ip_df):
        ip_feats = compute_ip_features(multi_user_ip_df)
        findings = _rules_ip(ip_feats)
        stuffing = [f for f in findings if f.category == "credential_stuffing"]
        assert len(stuffing) >= 1
        assert stuffing[0].entity_value == "192.168.2.99"

    def test_impossible_travel_detected(self):
        """User logging in from many IPs should trigger impossible_travel."""
        base = datetime(2026, 3, 27, 10, 0, 0)
        rows = [
            {
                "timestamp": (base + timedelta(minutes=i)).isoformat(sep=" "),
                "source_ip": f"10.0.{i}.1",
                "username":  "alice",
                "action":    "login_success",
            }
            for i in range(7)
        ]
        df         = _make_df(rows)
        user_feats = compute_user_features(df)
        findings   = _rules_user(user_feats)
        travel     = [f for f in findings if f.category == "impossible_travel"]
        assert len(travel) >= 1
        assert travel[0].entity_value == "alice"


# ── End-to-end pipeline test ───────────────────────────────────────────────────

class TestEndToEnd:
    def test_run_analysis_returns_list(self, clean_df):
        ip_feats, user_feats = compute_ip_features(clean_df), compute_user_features(clean_df)
        findings = run_analysis(ip_feats, user_feats)
        assert isinstance(findings, list)

    def test_findings_sorted_by_severity(self, brute_force_df):
        ip_feats   = compute_ip_features(brute_force_df)
        user_feats = compute_user_features(brute_force_df)
        findings   = run_analysis(ip_feats, user_feats)
        from src.app.analysis import SEVERITY_ORDER
        ranks = [SEVERITY_ORDER[f.severity] for f in findings]
        assert ranks == sorted(ranks), "Findings must be sorted by severity (most critical first)"

    def test_no_duplicate_findings(self, brute_force_df):
        ip_feats   = compute_ip_features(brute_force_df)
        user_feats = compute_user_features(brute_force_df)
        findings   = run_analysis(ip_feats, user_feats)
        keys = [(f.entity_type, f.entity_value, f.category) for f in findings]
        assert len(keys) == len(set(keys)), "Duplicate (entity, category) pairs found"
