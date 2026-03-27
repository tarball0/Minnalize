"""
NexusGuard Configuration
Central configuration for all system parameters.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    """Data generation and ingestion settings."""
    num_users: int = 50
    num_normal_events: int = 10_000
    num_anomalous_events: int = 500
    event_types: List[str] = field(default_factory=lambda: [
        "LOGIN", "LOGOUT", "FILE_ACCESS", "FILE_MODIFY", "FILE_DELETE",
        "NETWORK_CONNECT", "NETWORK_TRANSFER", "PRIVILEGE_ESCALATION",
        "CONFIG_CHANGE", "PROCESS_SPAWN", "REGISTRY_MODIFY", "USB_INSERT",
        "EMAIL_SEND", "DB_QUERY", "API_CALL",
    ])
    protocols: List[str] = field(default_factory=lambda: [
        "TCP", "UDP", "HTTP", "HTTPS", "SSH", "FTP", "DNS", "SMTP",
    ])
    severity_levels: List[str] = field(default_factory=lambda: [
        "INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL",
    ])


@dataclass
class FeatureConfig:
    """Feature engineering settings."""
    temporal_window_minutes: int = 60
    rolling_windows: List[int] = field(default_factory=lambda: [5, 15, 60, 360])
    behavioral_profile_lookback_hours: int = 24
    graph_min_edge_weight: int = 2
    sequence_length: int = 20


@dataclass
class ModelConfig:
    """Model hyperparameters."""
    # Autoencoder
    ae_encoding_dim: int = 16
    ae_hidden_layers: List[int] = field(default_factory=lambda: [64, 32])
    ae_epochs: int = 50
    ae_batch_size: int = 64
    ae_learning_rate: float = 1e-3
    ae_contamination: float = 0.10

    # LSTM
    lstm_hidden_size: int = 64
    lstm_num_layers: int = 2
    lstm_epochs: int = 30
    lstm_batch_size: int = 32
    lstm_learning_rate: float = 1e-3
    lstm_sequence_length: int = 20

    # Isolation Forest
    if_n_estimators: int = 200
    if_contamination: float = 0.10
    if_max_samples: str = "auto"

    # Ensemble
    ensemble_weights: dict = field(default_factory=lambda: {
        "autoencoder": 0.35,
        "lstm": 0.30,
        "isolation_forest": 0.20,
        "behavioral": 0.15,
    })
    anomaly_threshold: float = 0.50


@dataclass
class AlertConfig:
    """Alerting settings."""
    critical_threshold: float = 0.90
    high_threshold: float = 0.75
    medium_threshold: float = 0.60
    max_alerts_per_minute: int = 100


@dataclass
class NexusGuardConfig:
    """Master configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    alert: AlertConfig = field(default_factory=AlertConfig)
    random_seed: int = 42
    device: str = "cpu"  # "cpu" or "cuda"
