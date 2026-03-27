"""
LSTM Sequence Anomaly Detector
Captures temporal patterns in event sequences; detects unusual sequences.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from nexusguard.config import ModelConfig


class _LSTMNet(nn.Module):
    """LSTM network for sequence prediction/anomaly detection."""

    def __init__(self, input_dim: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden)

        # Self-attention over time steps
        attn_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(lstm_out * attn_weights, dim=1)  # (batch, hidden)

        output = self.fc(context)  # Predict next event features
        return output


class LSTMDetector:
    """LSTM-based sequence anomaly detector with attention mechanism."""

    def __init__(self, config: ModelConfig, input_dim: int, device: str = "cpu"):
        self.config = config
        self.device = torch.device(device)
        self.seq_len = config.lstm_sequence_length
        self.input_dim = input_dim
        self.model = _LSTMNet(
            input_dim=input_dim,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_num_layers,
        ).to(self.device)
        self.threshold = 0.0
        self._fitted = False

    def _create_sequences(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Create overlapping sequences for training."""
        sequences, targets = [], []
        for i in range(len(X) - self.seq_len):
            sequences.append(X[i:i + self.seq_len])
            targets.append(X[i + self.seq_len])
        return np.array(sequences), np.array(targets)

    def fit(self, X: np.ndarray) -> dict:
        """Train LSTM on normal sequences."""
        sequences, targets = self._create_sequences(X)
        if len(sequences) == 0:
            self._fitted = True
            return {"loss": []}

        dataset = TensorDataset(
            torch.FloatTensor(sequences),
            torch.FloatTensor(targets),
        )
        loader = DataLoader(dataset, batch_size=self.config.lstm_batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lstm_learning_rate)
        criterion = nn.MSELoss()

        self.model.train()
        history = {"loss": []}

        for epoch in range(self.config.lstm_epochs):
            epoch_loss = 0.0
            for batch_seq, batch_target in loader:
                batch_seq = batch_seq.to(self.device)
                batch_target = batch_target.to(self.device)

                predicted = self.model(batch_seq)
                loss = criterion(predicted, batch_target)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item() * len(batch_seq)

            avg_loss = epoch_loss / len(sequences)
            history["loss"].append(avg_loss)

        # Set threshold from training errors
        errors = self._compute_sequence_errors(X)
        if len(errors) > 0:
            self.threshold = np.percentile(errors, 95)
        self._fitted = True
        return history

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores for each event in the sequence."""
        errors = self._compute_sequence_errors(X)
        if len(errors) == 0:
            return np.zeros(len(X))

        # Pad the beginning (no sequence context available)
        padded = np.zeros(len(X))
        padded[self.seq_len:] = errors / (self.threshold * 2 + 1e-8)
        return np.clip(padded, 0, 1)

    def _compute_sequence_errors(self, X: np.ndarray) -> np.ndarray:
        """Compute prediction errors for sequences."""
        sequences, targets = self._create_sequences(X)
        if len(sequences) == 0:
            return np.array([])

        self.model.eval()
        with torch.no_grad():
            seq_tensor = torch.FloatTensor(sequences).to(self.device)
            target_tensor = torch.FloatTensor(targets).to(self.device)
            predicted = self.model(seq_tensor)
            errors = torch.mean((predicted - target_tensor) ** 2, dim=1).cpu().numpy()
        return errors
