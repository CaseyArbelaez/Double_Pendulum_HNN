from __future__ import annotations

import torch
from torch import nn


class BaselineVectorFieldNN(nn.Module):
    """Directly learns z -> z_dot."""

    def __init__(self, input_dim: int = 4, hidden_dim: int = 128, depth: int = 3):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, input_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
