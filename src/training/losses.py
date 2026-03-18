from __future__ import annotations

import torch


def baseline_vector_field_loss(model, z: torch.Tensor, z_dot_true: torch.Tensor) -> torch.Tensor:
    pred = model(z)
    return torch.mean((pred - z_dot_true) ** 2)


def hnn_vector_field_loss(model, z: torch.Tensor, z_dot_true: torch.Tensor) -> torch.Tensor:
    pred = model.time_derivative(z)
    return torch.mean((pred - z_dot_true) ** 2)
