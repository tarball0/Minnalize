#!/usr/bin/env python3
"""
NexusGuard - Main Entry Point
Run the full anomaly detection pipeline from the command line.

Usage:
    python main.py              # Run full pipeline with defaults
    python main.py --dashboard  # Launch Streamlit dashboard
"""

import argparse
import json
import logging
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nexusguard.config import NexusGuardConfig
from nexusguard.data import ActivityLogGenerator
from nexusguard.engine.pipeline import NexusGuardPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("nexusguard")


def run_pipeline(config: NexusGuardConfig) -> None:
    """Execute the full NexusGuard anomaly detection pipeline."""
    print("\n" + "=" * 70)
    print("   NexusGuard - Adaptive Behavioral Anomaly Detection Engine")
    print("=" * 70 + "\n")

    # 1. Generate data
    logger.info("Generating synthetic activity logs...")
    generator = ActivityLogGenerator(config)
    data = generator.generate()
    logger.info(f"Generated {len(data):,} events ({config.data.num_anomalous_events} injected anomalies)")
    logger.info(f"  Users: {config.data.num_users} | Time span: {data['timestamp'].min()} to {data['timestamp'].max()}")

    # 2. Train pipeline
    logger.info("\nTraining detection models...")
    pipeline = NexusGuardPipeline(config)
    train_metrics = pipeline.train(data)
    logger.info(f"Training metrics: {json.dumps(train_metrics, indent=2)}")

    # 3. Run detection
    logger.info("\nRunning anomaly detection...")
    events_df, scores_df, alerts = pipeline.detect_and_alert(data)

    # 4. Evaluate
    logger.info("\nEvaluating on labeled data...")
    eval_metrics = pipeline.evaluate(data)

    # 5. Display results
    print("\n" + "=" * 70)
    print("   DETECTION RESULTS")
    print("=" * 70)
    print(f"\n  Total Events Analyzed:    {len(events_df):,}")
    print(f"  Anomalies Detected:       {int(scores_df['is_anomaly'].sum()):,}")
    print(f"  Alerts Generated:         {len(alerts):,}")
    print(f"\n  {'─' * 40}")
    print(f"  EVALUATION METRICS")
    print(f"  {'─' * 40}")
    print(f"  Accuracy:                 {eval_metrics['accuracy']:.4f}")
    print(f"  Precision:                {eval_metrics['precision']:.4f}")
    print(f"  Recall:                   {eval_metrics['recall']:.4f}")
    print(f"  F1 Score:                 {eval_metrics['f1_score']:.4f}")
    print(f"  True Positives:           {eval_metrics['true_positives']}")
    print(f"  False Positives:          {eval_metrics['false_positives']}")
    print(f"  False Negatives:          {eval_metrics['false_negatives']}")
    print(f"  True Negatives:           {eval_metrics['true_negatives']}")

    # 6. Show top alerts
    if alerts:
        print(f"\n  {'─' * 40}")
        print(f"  TOP 10 ALERTS")
        print(f"  {'─' * 40}")
        sorted_alerts = sorted(alerts, key=lambda a: a.ensemble_score, reverse=True)[:10]
        for i, alert in enumerate(sorted_alerts, 1):
            print(f"\n  [{i}] {alert.alert_id} | {alert.risk_level:8s} | "
                  f"Score: {alert.ensemble_score:.3f}")
            print(f"      User: {alert.user_id} | Event: {alert.event_type}")
            print(f"      {alert.description}")

    # 7. Export alerts
    os.makedirs("output", exist_ok=True)
    pipeline.alert_engine.export_json("output/alerts.json")
    scores_df.to_csv("output/scores.csv", index=False)
    logger.info("\nAlerts exported to output/alerts.json")
    logger.info("Scores exported to output/scores.csv")

    print("\n" + "=" * 70)
    print("   Pipeline Complete")
    print("=" * 70 + "\n")


def launch_dashboard() -> None:
    """Launch the Streamlit dashboard."""
    import subprocess
    dashboard_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "nexusguard", "dashboard", "app.py",
    )
    subprocess.run([sys.executable, "-m", "streamlit", "run", dashboard_path,
                    "--server.headless", "true"])


def main():
    parser = argparse.ArgumentParser(description="NexusGuard Anomaly Detection System")
    parser.add_argument("--dashboard", action="store_true", help="Launch Streamlit dashboard")
    parser.add_argument("--events", type=int, default=5000, help="Number of normal events to generate")
    parser.add_argument("--anomalies", type=int, default=300, help="Number of anomalous events to inject")
    parser.add_argument("--users", type=int, default=50, help="Number of simulated users")
    parser.add_argument("--threshold", type=float, default=0.65, help="Anomaly detection threshold")

    args = parser.parse_args()

    if args.dashboard:
        launch_dashboard()
        return

    config = NexusGuardConfig()
    config.data.num_normal_events = args.events
    config.data.num_anomalous_events = args.anomalies
    config.data.num_users = args.users
    config.model.anomaly_threshold = args.threshold

    run_pipeline(config)


if __name__ == "__main__":
    main()
