import pandas as pd

def build_feature_matrix(df_raw, time_window='10min'):
    """Converts raw log events into a time-binned feature matrix."""
    if df_raw.empty:
        return pd.DataFrame()

    # 1. Ensure timestamp is a proper datetime object
    current_year = pd.Timestamp.now().year
    df_raw['timestamp'] = pd.to_datetime(f"{current_year} " + df_raw['timestamp'], format="%Y %b %d %H:%M:%S")

    # 2. Set timestamp as the index so we can slice it by time
    df_raw.set_index('timestamp', inplace=True)

    # 3. Resample into time buckets (e.g., every 10 mins)
    matrix = df_raw.resample(time_window).agg(
        total_events=('status', 'count'),
        failed_logins=('status', lambda x: (x == 'failed').sum()),
        success_logins=('status', lambda x: (x == 'success').sum()),
        unique_ips=('ip', 'nunique') # We still track how many unique IPs hit us in this window!
    ).reset_index()

    # 4. Add the hour of the day as a feature (2 AM is inherently more suspicious than 2 PM)
    matrix['hour_of_day'] = matrix['timestamp'].dt.hour

    # 5. Drop time windows where absolutely nothing happened
    matrix = matrix[matrix['total_events'] > 0].reset_index(drop=True)

    return matrix
