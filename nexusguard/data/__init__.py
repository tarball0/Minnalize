"""
Synthetic Activity Log Generator
Generates realistic system activity logs with injected anomalies.
"""

import random
from datetime import datetime, timedelta
from typing import List, Tuple

import numpy as np
import pandas as pd

from nexusguard.config import NexusGuardConfig


class ActivityLogGenerator:
    """Generates synthetic digital system activity logs with realistic patterns."""

    def __init__(self, config: NexusGuardConfig):
        self.cfg = config.data
        self.rng = np.random.default_rng(config.random_seed)
        random.seed(config.random_seed)

        self.users = [f"user_{i:03d}" for i in range(self.cfg.num_users)]
        self.hosts = [f"host-{i:02d}.internal" for i in range(15)]
        self.dest_ips = [f"10.0.{self.rng.integers(1, 255)}.{self.rng.integers(1, 255)}"
                         for _ in range(30)]
        self.external_ips = [f"{self.rng.integers(1, 223)}.{self.rng.integers(0, 255)}."
                             f"{self.rng.integers(0, 255)}.{self.rng.integers(1, 255)}"
                             for _ in range(20)]

    def _generate_normal_event(self, timestamp: datetime) -> dict:
        """Generate a single normal activity event."""
        user = self.rng.choice(self.users)
        event_type = self.rng.choice(self.cfg.event_types[:12])  # Normal events use first 12 types

        hour = timestamp.hour
        # Simulate work-hours bias
        if 9 <= hour <= 17:
            activity_rate = 1.0
        elif 6 <= hour <= 21:
            activity_rate = 0.4
        else:
            activity_rate = 0.1

        event = {
            "timestamp": timestamp,
            "user_id": user,
            "event_type": event_type,
            "source_host": self.rng.choice(self.hosts),
            "dest_ip": self.rng.choice(self.dest_ips),
            "protocol": self.rng.choice(self.cfg.protocols[:5]),  # Common protocols
            "port": int(self.rng.choice([22, 80, 443, 3306, 5432, 8080, 8443])),
            "bytes_sent": int(self.rng.exponential(500) * activity_rate),
            "bytes_received": int(self.rng.exponential(2000) * activity_rate),
            "duration_ms": int(self.rng.exponential(200)),
            "success": bool(self.rng.random() > 0.05),
            "severity": self.rng.choice(["INFO", "INFO", "INFO", "LOW", "LOW"]),
            "process_name": self.rng.choice([
                "chrome.exe", "outlook.exe", "python.exe", "code.exe",
                "explorer.exe", "svchost.exe", "bash", "java",
            ]),
            "is_anomaly": 0,
        }
        return event

    def _generate_anomalous_event(self, timestamp: datetime, attack_type: str) -> dict:
        """Generate an anomalous/malicious activity event."""
        user = self.rng.choice(self.users)

        anomaly_patterns = {
            "brute_force": {
                "event_type": "LOGIN",
                "success": False,
                "severity": "HIGH",
                "duration_ms": int(self.rng.integers(5, 50)),
                "bytes_sent": int(self.rng.integers(100, 300)),
                "process_name": "ssh",
                "port": 22,
            },
            "data_exfiltration": {
                "event_type": "NETWORK_TRANSFER",
                "success": True,
                "severity": "CRITICAL",
                "bytes_sent": int(self.rng.integers(500_000, 50_000_000)),
                "duration_ms": int(self.rng.integers(5000, 60000)),
                "process_name": self.rng.choice(["curl", "wget", "powershell.exe", "nc"]),
                "port": int(self.rng.choice([443, 8443, 4444, 9999])),
                "dest_ip": self.rng.choice(self.external_ips),
            },
            "privilege_escalation": {
                "event_type": "PRIVILEGE_ESCALATION",
                "success": True,
                "severity": "CRITICAL",
                "process_name": self.rng.choice(["sudo", "runas.exe", "psexec.exe"]),
                "port": 0,
            },
            "lateral_movement": {
                "event_type": "NETWORK_CONNECT",
                "success": True,
                "severity": "HIGH",
                "protocol": "SSH",
                "port": 22,
                "duration_ms": int(self.rng.integers(100, 2000)),
                "process_name": self.rng.choice(["ssh", "psexec.exe", "wmic.exe"]),
            },
            "off_hours_access": {
                "event_type": self.rng.choice(["FILE_ACCESS", "FILE_MODIFY", "DB_QUERY"]),
                "success": True,
                "severity": "MEDIUM",
                "duration_ms": int(self.rng.integers(500, 5000)),
            },
            "config_tampering": {
                "event_type": self.rng.choice(["CONFIG_CHANGE", "REGISTRY_MODIFY"]),
                "success": True,
                "severity": "HIGH",
                "process_name": self.rng.choice(["regedit.exe", "sed", "vi"]),
            },
        }

        pattern = anomaly_patterns[attack_type]

        # Force off-hours for off_hours_access
        if attack_type == "off_hours_access":
            timestamp = timestamp.replace(hour=self.rng.integers(0, 5))

        event = {
            "timestamp": timestamp,
            "user_id": user,
            "event_type": pattern.get("event_type", "PROCESS_SPAWN"),
            "source_host": self.rng.choice(self.hosts),
            "dest_ip": pattern.get("dest_ip", self.rng.choice(self.dest_ips)),
            "protocol": pattern.get("protocol", self.rng.choice(self.cfg.protocols)),
            "port": pattern.get("port", int(self.rng.choice([22, 80, 443, 4444]))),
            "bytes_sent": pattern.get("bytes_sent", int(self.rng.exponential(1000))),
            "bytes_received": int(self.rng.exponential(2000)),
            "duration_ms": pattern.get("duration_ms", int(self.rng.exponential(500))),
            "success": pattern.get("success", True),
            "severity": pattern.get("severity", "HIGH"),
            "process_name": pattern.get("process_name", "unknown"),
            "is_anomaly": 1,
            "attack_type": attack_type,
        }
        return event

    def generate(self) -> pd.DataFrame:
        """Generate full dataset with normal and anomalous events."""
        events = []
        base_time = datetime(2026, 3, 1, 0, 0, 0)

        # Generate normal events spread over ~7 days
        for _ in range(self.cfg.num_normal_events):
            offset = timedelta(
                days=int(self.rng.integers(0, 7)),
                hours=int(self.rng.choice(
                    range(24), p=self._work_hour_distribution()
                )),
                minutes=int(self.rng.integers(0, 60)),
                seconds=int(self.rng.integers(0, 60)),
            )
            events.append(self._generate_normal_event(base_time + offset))

        # Generate anomalous events
        attack_types = ["brute_force", "data_exfiltration", "privilege_escalation",
                        "lateral_movement", "off_hours_access", "config_tampering"]
        for _ in range(self.cfg.num_anomalous_events):
            offset = timedelta(
                days=int(self.rng.integers(0, 7)),
                hours=int(self.rng.integers(0, 24)),
                minutes=int(self.rng.integers(0, 60)),
                seconds=int(self.rng.integers(0, 60)),
            )
            attack = self.rng.choice(attack_types)
            events.append(self._generate_anomalous_event(base_time + offset, attack))

        df = pd.DataFrame(events)
        df = df.sort_values("timestamp").reset_index(drop=True)
        # Fill missing attack_type for normal events
        if "attack_type" not in df.columns:
            df["attack_type"] = "none"
        df["attack_type"] = df["attack_type"].fillna("none")
        return df

    def _work_hour_distribution(self) -> np.ndarray:
        """Generate probability distribution favoring work hours."""
        probs = np.array([
            0.01, 0.01, 0.01, 0.01, 0.01, 0.02,  # 0-5
            0.03, 0.05, 0.08, 0.10, 0.10, 0.10,  # 6-11
            0.08, 0.10, 0.10, 0.08, 0.06, 0.04,  # 12-17
            0.03, 0.02, 0.02, 0.01, 0.01, 0.01,  # 18-23
        ])
        return probs / probs.sum()
