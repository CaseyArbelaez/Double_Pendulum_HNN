from __future__ import annotations

import torch
from torch import nn


class HamiltonianNN(nn.Module):
    """Learns scalar H(z), then induces z_dot = J grad H(z)."""

    def __init__(self, input_dim: int = 4, hidden_dim: int = 128, depth: int = 3):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

        J = torch.tensor(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        self.register_buffer("J", J)

    def hamiltonian(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)

    def time_derivative(self, z: torch.Tensor) -> torch.Tensor:
        if not z.requires_grad:
            z = z.clone().detach().requires_grad_(True)
        H = self.hamiltonian(z)
        gradH = torch.autograd.grad(H.sum(), z, create_graph=True)[0]
        return gradH @ self.J.T

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.time_derivative(z)
