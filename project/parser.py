import pandas as pd
import re
from datetime import datetime

def parse_auth_log(filepath="data/auth.log"):
    """Reads raw Linux auth.log and extracts timestamp, IP, and status."""
    data = []
    # Standard syslog regex: Month Day Time Host Process: Message
    log_pattern = re.compile(r"([A-Z][a-z]{2}\s+\d+\s\d{2}:\d{2}:\d{2})\s+\w+\s+sshd\[\d+\]:\s+(.*)")
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                match = log_pattern.search(line)
                if match:
                    timestamp_str, message = match.groups()
                    
                    # Extract IP
                    ip_match = re.search(r"from\s+([0-9\.]+)", message)
                    ip = ip_match.group(1) if ip_match else "unknown"
                    
                    # Determine Status
                    status = "failed" if "Failed" in message else "success" if "Accepted" in message else "other"
                    
                    data.append({
                        "timestamp": timestamp_str, # Will parse properly in features.py
                        "ip": ip,
                        "status": status
                    })
    except FileNotFoundError:
        pass # Handle gracefully in dashboard

    return pd.DataFrame(data)

