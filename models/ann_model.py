import logging
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("FingerprintANN")


@dataclass
class TrainingConfig:
    learning_rate: float = 0.001
    epochs: int = 150
    batch_size: int = 32
    weight_decay: float = 1e-4
    dropout: float = 0.3
    early_stopping_patience: int = 15
    lr_step_size: int = 30
    lr_gamma: float = 0.5


class FingerprintANN(nn.Module):
    """
    ANN: 1764 → 512 → 256 → 10
    Matches the architecture from the thesis (HOG + ANN).
    BatchNorm + Dropout added for regularization.
    """

    def __init__(
        self,
        input_dim: int = 1764,
        hidden_layers: list[int] = None,
        output_dim: int = 10,
        dropout: float = 0.3,
    ):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [512, 256]

        layers = []
        prev = input_dim
        for size in hidden_layers:
            layers += [
                nn.Linear(prev, size),
                nn.BatchNorm1d(size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            prev = size

        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict(self, x: torch.Tensor) -> tuple[float, int]:
        """
        Single-sample inference.
        Returns (confidence, class_id) matching the interface of onnx_inference.py.
        """
        self.eval()
        with torch.no_grad():
            if x.dim() == 1:
                x = x.unsqueeze(0)
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
            confidence = probs.max().item()
            class_id = probs.argmax().item()
        return confidence, class_id
