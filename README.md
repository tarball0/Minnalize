# NEXUS 2026 — Activity Pattern Analysis & Threat Detection

A Python-based system that ingests activity logs, builds behavioural baselines,
and surfaces unusual or suspicious behaviour using three complementary detection
layers.

---

## Project Structure

```
nexus_2026/
├── data/
│   ├── generate_logs.py     # Synthetic log generator (run this first)
│   └── activity.log         # Generated after running the generator
├── src/
│   └── app/
│       ├── ingestion.py     # Load & validate CSV logs → DataFrame
│       ├── processing.py    # Feature engineering (per-IP, per-user)
│       ├── analysis.py      # Three-layer anomaly detection engine
│       ├── alerting.py      # Terminal report + JSON export
│       └── main.py          # CLI entry point
├── tests/
│   └── test_analysis.py     # pytest suite
├── requirements.txt
└── README.md
```

---

## How It Works

### 1. Data Ingestion (`ingestion.py`)
Reads a CSV log with four columns: `timestamp`, `source_ip`, `username`, `action`.
Validates the schema, strips whitespace, drops nulls, and returns a sorted
`pandas.DataFrame`.

### 2. Feature Engineering (`processing.py`)
Aggregates raw events into two behavioural profiles:

| Dimension | Features computed |
|-----------|------------------|
| **Per-IP** | total events, failure count/rate, unique usernames, sudo count, after-hours count, event rate per minute |
| **Per-User** | total events, failure count/rate, unique IPs, sudo count, after-hours count |

### 3. Analysis Engine (`analysis.py`) — Three Layers

| Layer | Technique | What it catches |
|-------|-----------|-----------------|
| **Rules** | Hard thresholds | Brute force (≥10 failures), high failure rate (≥75%), credential stuffing (IP → many users), impossible travel (user → many IPs), after-hours activity |
| **Z-score** | Statistical baseline | Any feature that deviates ≥2.5σ from the population mean — adapts as the log grows |
| **Isolation Forest** | Unsupervised ML | Multi-dimensional outliers that rules/z-score may miss; trains fresh on every run |

Findings from all three layers are merged, deduplicated (same entity + category keeps
the highest severity), and sorted CRITICAL → HIGH → MEDIUM → LOW.

### 4. Alerting (`alerting.py`)
Prints a colour-coded terminal report and optionally writes a machine-readable
JSON file suitable for SIEM ingestion.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic log data (500 normal events + injected anomalies)
python data/generate_logs.py --rows 500 --out data/activity.log

# 3. Run the analyser
python -m src.app.main --log data/activity.log

# 4. (Optional) Save a JSON report
python -m src.app.main --log data/activity.log --json-out reports/report.json

# 5. Filter to high-severity findings only
python -m src.app.main --log data/activity.log --min-severity HIGH
```

### Running Tests

```bash
pytest tests/ -v
```

---

## Injected Anomalies (in the generated data)

The synthetic log generator seeds three distinct threat patterns:

| Pattern | Description |
|---------|-------------|
| **Brute force** | 40 rapid `login_failure` events from `10.99.0.13` between 1–5 AM, followed by a successful login as `scanner` |
| **After-hours insider** | `alice` logs in at 03:14 AM from `192.168.1.77` (an unknown internal host) and immediately runs `sudo` |
| **Impossible travel** | `bob` authenticates from 10 brand-new `172.16.x.x` IPs within 2.5 minutes |

---

## Tuning Thresholds

All thresholds are centralised at the top of `src/app/analysis.py`:

```python
BRUTE_FORCE_FAILURE_THRESHOLD  = 10    # ≥ N failures from one IP  → CRITICAL
HIGH_FAILURE_RATE_THRESHOLD    = 0.75  # ≥ 75% failures             → HIGH
AFTER_HOURS_THRESHOLD          = 2     # ≥ N after-hours events     → MEDIUM
UNIQUE_USERS_PER_IP_THRESHOLD  = 4     # one IP → many users        → HIGH
UNIQUE_IPS_PER_USER_THRESHOLD  = 5     # one user → many IPs        → HIGH
ZSCORE_THRESHOLD               = 2.5   # standard deviations        → MEDIUM
IF_CONTAMINATION               = 0.05  # assumed anomaly rate (5%)
```

Adjust these to match your environment's risk appetite and traffic volume.
