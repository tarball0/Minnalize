"""
Deep Autoencoder Anomaly Detector
Learns compressed representations of normal behavior; high reconstruction error signals anomalies.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from nexusguard.config import ModelConfig


class _AutoencoderNet(nn.Module):
    """Symmetric deep autoencoder network."""

    def __init__(self, input_dim: int, hidden_layers: list[int], encoding_dim: int):
        super().__init__()

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_layers:
            encoder_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_layers.append(nn.BatchNorm1d(h_dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(0.2))
            prev_dim = h_dim
        encoder_layers.append(nn.Linear(prev_dim, encoding_dim))
        encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder (mirror)
        decoder_layers = []
        prev_dim = encoding_dim
        for h_dim in reversed(hidden_layers):
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(nn.BatchNorm1d(h_dim))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Dropout(0.2))
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded


class AutoencoderDetector:
    """Anomaly detector based on autoencoder reconstruction error."""

    def __init__(self, config: ModelConfig, input_dim: int, device: str = "cpu"):
        self.config = config
        self.device = torch.device(device)
        self.model = _AutoencoderNet(
            input_dim=input_dim,
            hidden_layers=config.ae_hidden_layers,
            encoding_dim=config.ae_encoding_dim,
        ).to(self.device)
        self.threshold = 0.0
        self._fitted = False

    def fit(self, X: np.ndarray) -> dict:
        """Train autoencoder on normal data."""
        dataset = TensorDataset(torch.FloatTensor(X))
        loader = DataLoader(dataset, batch_size=self.config.ae_batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.ae_learning_rate)
        criterion = nn.MSELoss()

        self.model.train()
        history = {"loss": []}

        for epoch in range(self.config.ae_epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(self.device)
                reconstructed, _ = self.model(batch)
                loss = criterion(reconstructed, batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(batch)

            avg_loss = epoch_loss / len(X)
            history["loss"].append(avg_loss)

        # Compute threshold from training data reconstruction errors
        errors = self._compute_errors(X)
        self.threshold = np.percentile(errors, (1 - self.config.ae_contamination) * 100)
        self._fitted = True
        return history

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores (normalized reconstruction errors)."""
        errors = self._compute_errors(X)
        # Normalize to [0, 1] using the threshold
        scores = errors / (self.threshold * 2 + 1e-8)
        return np.clip(scores, 0, 1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary predictions (1 = anomaly)."""
        errors = self._compute_errors(X)
        return (errors > self.threshold).astype(int)

    def _compute_errors(self, X: np.ndarray) -> np.ndarray:
        """Compute per-sample reconstruction errors."""
        self.model.eval()
        with torch.no_grad():
            data = torch.FloatTensor(X).to(self.device)
            reconstructed, _ = self.model(data)
            errors = torch.mean((data - reconstructed) ** 2, dim=1).cpu().numpy()
        return errors

    def get_encodings(self, X: np.ndarray) -> np.ndarray:
        """Get the latent space encodings for visualization."""
        self.model.eval()
        with torch.no_grad():
            data = torch.FloatTensor(X).to(self.device)
            _, encoded = self.model(data)
        return encoded.cpu().numpy()
