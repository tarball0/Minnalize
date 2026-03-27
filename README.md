# NexusGuard — Adaptive Behavioral Anomaly Detection Engine

An advanced anomaly detection system that analyzes digital system activity logs to surface suspicious behavior using an ensemble of deep learning, sequence modeling, statistical, and graph-based techniques.

## Architecture

```
nexusguard/
├── config.py                    # Central configuration
├── data/
│   ├── __init__.py              # Synthetic activity log generator
│   └── preprocessor.py          # Data cleaning, encoding, normalization
├── features/
│   ├── temporal.py              # Rolling-window temporal features
│   ├── behavioral.py            # Per-user behavioral profiling
│   └── graph_features.py        # Graph-based communication analysis
├── models/
│   ├── autoencoder.py           # Deep Autoencoder (reconstruction error)
│   ├── lstm_detector.py         # LSTM + Attention (sequence anomalies)
│   ├── isolation_forest.py      # Isolation Forest (statistical outliers)
│   └── ensemble.py              # Adaptive ensemble meta-scorer
├── engine/
│   ├── pipeline.py              # End-to-end orchestration
│   └── alerter.py               # Alert generation and reporting
└── dashboard/
    └── app.py                   # Interactive Streamlit dashboard
```

## Advanced Techniques Used

| Component | Technique | Purpose |
|-----------|-----------|---------|
| **Deep Autoencoder** | Neural network with encoder-decoder + BatchNorm + Dropout | Learns compressed representations of normal behavior; high reconstruction error = anomaly |
| **LSTM + Self-Attention** | Recurrent network with attention over time steps | Captures temporal ordering patterns; detects unusual event sequences |
| **Isolation Forest** | Ensemble of random trees | Isolates outliers in high-dimensional feature space via random partitioning |
| **Behavioral Profiler** | Per-user statistical models with z-scores | Flags deviations from individual baseline patterns (hours, data volume, destinations) |
| **Graph Analyzer** | NetworkX community detection + edge novelty | Detects unusual communication patterns and new entity relationships |
| **Ensemble Meta-Scorer** | Weighted fusion + agreement bonus + adaptive reweighting | Combines all models with dynamic weights that adapt based on feedback |
| **Temporal Features** | Multi-scale rolling windows (5m, 15m, 1h, 6h) | Captures short and long-term activity patterns, burst detection |

## Quick Start

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run CLI Pipeline
```bash
python main.py --events 5000 --anomalies 300 --users 50
```

### Launch Interactive Dashboard
```bash
python main.py --dashboard
# OR directly:
streamlit run nexusguard/dashboard/app.py
```

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--dashboard` | off | Launch Streamlit UI |
| `--events` | 5000 | Number of normal events to simulate |
| `--anomalies` | 300 | Number of injected attack events |
| `--users` | 50 | Number of simulated user accounts |
| `--threshold` | 0.65 | Ensemble anomaly score threshold |

## Attack Types Simulated

- **Brute Force** — Rapid failed login attempts
- **Data Exfiltration** — Large outbound data transfers to external IPs
- **Privilege Escalation** — Unauthorized privilege elevation
- **Lateral Movement** — Internal network traversal via SSH/remote tools
- **Off-Hours Access** — File/DB access during unusual hours
- **Config Tampering** — Unauthorized system configuration changes

## Dashboard Features

1. **Overview** — KPIs, score distribution, risk pie chart, activity timeline
2. **Anomaly Explorer** — Interactive scatter plots, filterable event table
3. **Alerts** — Severity-sorted alert cards with model score breakdowns
4. **Model Performance** — Confusion matrix, per-model score distributions, training metrics
5. **Deep Dive** — Top risky users, event-hour risk heatmap, t-SNE latent space visualization

## How the Ensemble Works

Each event gets scored by all four models independently (0–1 scale). The ensemble combines scores with configurable weights and adds an **agreement bonus** when 3+ models flag the same event:

```
final_score = Σ(weight_i × score_i) + agreement_bonus
```

The system also supports **adaptive weight reweighting** — when ground-truth feedback is available, model weights are adjusted based on each model's F1 score, so the ensemble improves over time.

## Team

**Team MinnalManaf** — ACM Nexus 2026
