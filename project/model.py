from sklearn.ensemble import IsolationForest

def detect_anomalies(df_matrix, contamination=0.05):
    """Runs Isolation Forest on the feature matrix."""
    if df_matrix.empty:
        return df_matrix

    # FIX: Updated to match features.py and include unique_ips
    features = ['total_events', 'failed_logins', 'success_logins', 'unique_ips', 'hour_of_day']
    X = df_matrix[features]

    # Train and predict
    model = IsolationForest(contamination=contamination, random_state=42)
    df_matrix['anomaly_score'] = model.fit_predict(X)
    
    return df_matrix
