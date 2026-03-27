"""
ingestion.py
------------
Loads and validates the activity log CSV into a pandas DataFrame.

Expected columns: timestamp, source_ip, username, action
"""

import pandas as pd
from pathlib import Path


REQUIRED_COLUMNS = {"timestamp", "source_ip", "username", "action"}

VALID_ACTIONS = {
    "login_success",
    "login_failure",
    "sudo",
    "logout",
    "new_device",
}


def load_log(file_path: str) -> pd.DataFrame:
    """
    Parse a CSV activity log and return a clean DataFrame.

    Raises:
        FileNotFoundError – if the path does not exist.
        ValueError        – if required columns are missing.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {file_path}")

    df = pd.read_csv(path, parse_dates=["timestamp"])

    # Validate schema
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Log file is missing columns: {missing}")

    # Strip whitespace from string columns
    for col in ("source_ip", "username", "action"):
        df[col] = df[col].str.strip()

    # Drop rows with null critical fields
    before = len(df)
    df = df.dropna(subset=list(REQUIRED_COLUMNS))
    dropped = before - len(df)
    if dropped:
        print(f"[!] Dropped {dropped} rows with missing values.")

    # Warn about unknown actions (don't drop – future-proofing)
    unknown = set(df["action"].unique()) - VALID_ACTIONS
    if unknown:
        print(f"[!] Unknown action types encountered (will be kept): {unknown}")

    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"[+] Loaded {len(df)} events from {file_path}")
    return df
