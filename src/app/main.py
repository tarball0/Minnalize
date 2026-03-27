"""
main.py
-------
CLI entry point for the NEXUS 2026 activity-analysis system.

Usage examples:
  # Generate sample data then analyse it
  python -m src.app.main --log data/activity.log

  # Save a JSON report alongside the terminal output
  python -m src.app.main --log data/activity.log --json-out reports/report.json

  # Adjust the minimum severity shown
  python -m src.app.main --log data/activity.log --min-severity HIGH
"""

import argparse
import sys
from pathlib import Path

from .ingestion  import load_log
from .processing import build_features
from .analysis   import run_analysis, SEVERITY_ORDER
from .alerting   import print_report, save_json_report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="nexus",
        description="NEXUS 2026 – Activity pattern analysis and threat detection",
    )
    parser.add_argument(
        "--log", "-l",
        required=True,
        metavar="FILE",
        help="Path to the CSV activity log to analyse.",
    )
    parser.add_argument(
        "--json-out", "-j",
        metavar="FILE",
        default=None,
        help="Optional path to write a JSON report.",
    )
    parser.add_argument(
        "--min-severity", "-s",
        choices=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
        default="LOW",
        help="Minimum severity level to include in the report (default: LOW).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # ── Step 1: Ingest ────────────────────────────────────────────────────────
    try:
        df = load_log(args.log)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    if df.empty:
        print("[ERROR] Log file contained no usable events.", file=sys.stderr)
        return 1

    # ── Step 2: Feature engineering ───────────────────────────────────────────
    ip_feats, user_feats = build_features(df)
    print(f"[+] Feature tables built — {len(ip_feats)} IPs, {len(user_feats)} users.")

    # ── Step 3: Analyse ───────────────────────────────────────────────────────
    findings = run_analysis(ip_feats, user_feats)
    print(f"[+] Analysis complete — {len(findings)} finding(s) before severity filter.")

    # ── Step 4: Filter by minimum severity ───────────────────────────────────
    min_rank = SEVERITY_ORDER[args.min_severity]
    findings = [f for f in findings if SEVERITY_ORDER[f.severity] <= min_rank]

    # ── Step 5: Report ────────────────────────────────────────────────────────
    print_report(findings, log_file=args.log)

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        save_json_report(findings, args.json_out)

    # Exit code: 0 = clean, 2 = findings present (useful in CI pipelines)
    return 2 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
