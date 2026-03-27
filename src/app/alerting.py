"""
alerting.py
-----------
Formats and prints Finding objects to the terminal with ANSI colour coding.

Severity colour mapping:
  CRITICAL → bold red
  HIGH     → red
  MEDIUM   → yellow
  LOW      → cyan

Optionally writes findings to a JSON report file.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .analysis import Finding

# ── ANSI colour codes ──────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"

COLOURS = {
    "CRITICAL": f"{BOLD}\033[91m",   # bold bright-red
    "HIGH":     "\033[91m",          # bright-red
    "MEDIUM":   "\033[93m",          # yellow
    "LOW":      "\033[96m",          # cyan
}

SEVERITY_ICONS = {
    "CRITICAL": "🔴",
    "HIGH":     "🟠",
    "MEDIUM":   "🟡",
    "LOW":      "🔵",
}


def _colour(severity: str, text: str) -> str:
    return f"{COLOURS.get(severity, '')}{text}{RESET}"


def _separator(char: str = "─", width: int = 72) -> str:
    return char * width


# ── Public functions ───────────────────────────────────────────────────────────

def print_report(findings: list, log_file: str = "") -> None:
    """
    Print a formatted threat report to stdout.

    Args:
        findings:  Sorted list of Finding objects from analysis.run_analysis().
        log_file:  Source log path (for the header).
    """
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    print()
    print(_separator("═"))
    print(f"  NEXUS 2026 — Threat Intelligence Report")
    print(f"  Generated : {ts}")
    if log_file:
        print(f"  Source    : {log_file}")
    print(_separator("═"))

    if not findings:
        print("\n  ✅  No suspicious activity detected.\n")
        print(_separator("═"))
        return

    # Group by severity for the summary header
    counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for f in findings:
        counts[f.severity] = counts.get(f.severity, 0) + 1

    print("\n  SUMMARY")
    for sev in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
        if counts[sev]:
            icon = SEVERITY_ICONS[sev]
            print(f"    {icon}  {_colour(sev, sev):<30s}  {counts[sev]} finding(s)")
    print()
    print(_separator())

    # Individual findings
    for i, f in enumerate(findings, start=1):
        icon   = SEVERITY_ICONS.get(f.severity, "")
        header = _colour(f.severity, f"[{f.severity}]")
        entity = f"{'IP' if f.entity_type == 'ip' else 'User':4s}  {f.entity_value}"
        print(f"\n  #{i:03d}  {icon} {header}  {entity}")
        print(f"        Category : {f.category}")
        print(f"        Layer    : {f.layer}")
        print(f"        Detail   : {f.description}")
        if f.evidence:
            ev_str = "  |  ".join(f"{k}: {v}" for k, v in f.evidence.items())
            print(f"        Evidence : {ev_str}")
        print(_separator("·"))

    print()
    print(_separator("═"))
    print(f"  Total findings: {len(findings)}")
    print(_separator("═"))
    print()


def save_json_report(findings: list, output_path: str) -> None:
    """
    Serialise all findings to a JSON file for downstream consumption
    (SIEM ingestion, dashboards, etc.).
    """
    records = []
    for f in findings:
        records.append({
            "entity_type":  f.entity_type,
            "entity_value": f.entity_value,
            "severity":     f.severity,
            "category":     f.category,
            "description":  f.description,
            "layer":        f.layer,
            "evidence":     f.evidence,
        })

    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "total_findings": len(records),
        "findings": records,
    }

    with open(output_path, "w") as fh:
        json.dump(report, fh, indent=2)

    print(f"[+] JSON report saved → {output_path}")
