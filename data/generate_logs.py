"""
generate_logs.py
----------------
Generates synthetic auth activity logs (CSV format) with realistic normal
behaviour and deliberately injected anomalies.

Output columns:
  timestamp   – ISO‑8601 datetime string
  source_ip   – IPv4 address of the connecting host
  username    – account name
  action      – login_success | login_failure | sudo | logout | new_device

Usage:
  python data/generate_logs.py          # writes data/activity.log
  python data/generate_logs.py --rows 5000 --out data/big.log
"""

import csv
import random
import argparse
from datetime import datetime, timedelta

# ── Configuration ─────────────────────────────────────────────────────────────

NORMAL_USERS = ["alice", "bob", "carol", "dave", "eve"]
NORMAL_IPS   = [f"192.168.1.{i}" for i in range(10, 25)]
ACTIONS      = ["login_success", "login_failure", "sudo", "logout"]

# Anomaly seeds – these will stand out clearly in the report
ANOMALY_ATTACKER_IP  = "10.99.0.13"   # external attacker
ANOMALY_ROGUE_USER   = "scanner"      # never‑seen‑before username
ANOMALY_INSIDER_IP   = "192.168.1.77" # inside net but unknown host

BUSINESS_HOURS = range(8, 18)         # 08:00 – 17:59 = "normal" window


def random_timestamp(base: datetime, jitter_minutes: int = 60) -> datetime:
    return base + timedelta(minutes=random.randint(0, jitter_minutes))


def generate_normal_events(base: datetime, n: int) -> list[dict]:
    events = []
    for _ in range(n):
        hour   = random.choice(list(BUSINESS_HOURS))
        minute = random.randint(0, 59)
        ts     = base.replace(hour=hour, minute=minute, second=random.randint(0, 59))
        events.append({
            "timestamp": ts.isoformat(sep=" "),
            "source_ip": random.choice(NORMAL_IPS),
            "username":  random.choice(NORMAL_USERS),
            "action":    random.choices(ACTIONS, weights=[60, 5, 20, 15])[0],
        })
    return events


def generate_brute_force(base: datetime, n_attempts: int = 40) -> list[dict]:
    """Many rapid login failures from the same external IP."""
    events = []
    start = base.replace(hour=random.randint(1, 5), minute=0, second=0)
    for i in range(n_attempts):
        ts = start + timedelta(seconds=i * 3)   # one attempt every 3 s
        events.append({
            "timestamp": ts.isoformat(sep=" "),
            "source_ip": ANOMALY_ATTACKER_IP,
            "username":  random.choice(NORMAL_USERS + [ANOMALY_ROGUE_USER]),
            "action":    "login_failure",
        })
    # Attacker eventually succeeds
    events.append({
        "timestamp": (start + timedelta(seconds=n_attempts * 3)).isoformat(sep=" "),
        "source_ip": ANOMALY_ATTACKER_IP,
        "username":  ANOMALY_ROGUE_USER,
        "action":    "login_success",
    })
    return events


def generate_after_hours_insider(base: datetime) -> list[dict]:
    """Legitimate user logs in at 3 AM from an unknown internal host."""
    ts = base.replace(hour=3, minute=14, second=7)
    return [
        {"timestamp": ts.isoformat(sep=" "), "source_ip": ANOMALY_INSIDER_IP,
         "username": "alice", "action": "login_success"},
        {"timestamp": (ts + timedelta(minutes=2)).isoformat(sep=" "),
         "source_ip": ANOMALY_INSIDER_IP, "username": "alice", "action": "sudo"},
        {"timestamp": (ts + timedelta(minutes=5)).isoformat(sep=" "),
         "source_ip": ANOMALY_INSIDER_IP, "username": "alice", "action": "logout"},
    ]


def generate_new_device_spike(base: datetime) -> list[dict]:
    """A normal user connects from 10 brand‑new IPs within minutes."""
    ts  = base.replace(hour=9, minute=30, second=0)
    ips = [f"172.16.{random.randint(0, 255)}.{random.randint(1, 254)}" for _ in range(10)]
    return [
        {"timestamp": (ts + timedelta(seconds=i * 15)).isoformat(sep=" "),
         "source_ip": ip, "username": "bob", "action": "login_success"}
        for i, ip in enumerate(ips)
    ]


def build_dataset(n_normal: int, base: datetime) -> list[dict]:
    rows = generate_normal_events(base, n_normal)
    rows += generate_brute_force(base)
    rows += generate_after_hours_insider(base)
    rows += generate_new_device_spike(base)
    # Sort chronologically then add a small random jitter so events interleave
    rows.sort(key=lambda r: r["timestamp"])
    return rows


def write_csv(rows: list[dict], path: str) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["timestamp", "source_ip", "username", "action"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[+] Wrote {len(rows)} events → {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic activity logs")
    parser.add_argument("--rows", type=int, default=500, help="Number of normal events (default 500)")
    parser.add_argument("--out",  type=str, default="data/activity.log", help="Output file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)
    base_date = datetime(2026, 3, 27)
    dataset   = build_dataset(args.rows, base_date)
    write_csv(dataset, args.out)
