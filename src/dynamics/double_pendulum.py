from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch


@dataclass
class DoublePendulumHamiltonian:
    """
    Exact double pendulum in canonical coordinates.

    State ordering:
        z = [theta1, theta2, p1, p2]

    Hamiltonian:
        H(theta1, theta2, p1, p2)

    Dynamics:
        z_dot = J grad H(z)

    Notes
    -----
    - This class supports both NumPy and PyTorch workflows.
    - The PyTorch path is used for autograd-based derivatives.
    - The NumPy path is used for lightweight simulation and dataset generation.
    """

    m1: float = 1.0
    m2: float = 1.0
    l1: float = 1.0
    l2: float = 1.0
    g: float = 9.81

    @property
    def J(self) -> np.ndarray:
        return np.array(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

    def hamiltonian_torch(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute H(z) for a batch of states using PyTorch.

        Parameters
        ----------
        z : torch.Tensor
            Shape (..., 4)

        Returns
        -------
        torch.Tensor
            Shape (...,)
        """
        theta1, theta2, p1, p2 = torch.unbind(z, dim=-1)
        delta = theta1 - theta2

        m1 = self.m1
        m2 = self.m2
        l1 = self.l1
        l2 = self.l2
        g = self.g

        denom = 2.0 * m2 * l1**2 * l2**2 * (m1 + m2 * torch.sin(delta) ** 2)
        numer = (
            m2 * l2**2 * p1**2
            + (m1 + m2) * l1**2 * p2**2
            - 2.0 * m2 * l1 * l2 * torch.cos(delta) * p1 * p2
        )

        kinetic = numer / denom
        potential = -(m1 + m2) * g * l1 * torch.cos(theta1) - m2 * g * l2 * torch.cos(theta2)
        return kinetic + potential

    def hamiltonian_numpy(self, z: np.ndarray) -> np.ndarray:
        """
        NumPy version of the Hamiltonian.

        Parameters
        ----------
        z : np.ndarray
            Shape (..., 4)

        Returns
        -------
        np.ndarray
            Shape (...,)
        """
        theta1 = z[..., 0]
        theta2 = z[..., 1]
        p1 = z[..., 2]
        p2 = z[..., 3]
        delta = theta1 - theta2

        m1 = self.m1
        m2 = self.m2
        l1 = self.l1
        l2 = self.l2
        g = self.g

        denom = 2.0 * m2 * l1**2 * l2**2 * (m1 + m2 * np.sin(delta) ** 2)
        numer = (
            m2 * l2**2 * p1**2
            + (m1 + m2) * l1**2 * p2**2
            - 2.0 * m2 * l1 * l2 * np.cos(delta) * p1 * p2
        )

        kinetic = numer / denom
        potential = -(m1 + m2) * g * l1 * np.cos(theta1) - m2 * g * l2 * np.cos(theta2)
        return kinetic + potential

    def vector_field_torch(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute z_dot = J grad H(z) using autograd.

        Parameters
        ----------
        z : torch.Tensor
            Shape (batch, 4)

        Returns
        -------
        torch.Tensor
            Shape (batch, 4)
        """
        if not z.requires_grad:
            z = z.clone().detach().requires_grad_(True)

        H = self.hamiltonian_torch(z)
        gradH = torch.autograd.grad(H.sum(), z, create_graph=True)[0]
        J = torch.tensor(self.J, dtype=z.dtype, device=z.device)
        return gradH @ J.T

    def vector_field_numpy(self, z: np.ndarray) -> np.ndarray:
        """
        NumPy vector field via torch autograd on a single state or batch.

        This is a convenience wrapper used during dataset generation.
        It is not the most optimized path, but it is compact and reliable.
        """
        z_t = torch.tensor(z, dtype=torch.float64, requires_grad=True)
        batched = z_t.ndim == 2
        if not batched:
            z_t = z_t.unsqueeze(0)

        z_dot = self.vector_field_torch(z_t).detach().cpu().numpy()
        return z_dot if batched else z_dot[0]

    def energy(self, z: np.ndarray) -> np.ndarray:
        return self.hamiltonian_numpy(z)

    def sample_initial_state(
        self,
        theta1_range: Tuple[float, float] = (-1.0, 1.0),
        theta2_range: Tuple[float, float] = (-1.0, 1.0),
        p1_range: Tuple[float, float] = (-0.5, 0.5),
        p2_range: Tuple[float, float] = (-0.5, 0.5),
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Sample a random initial state in canonical coordinates."""
        rng = np.random.default_rng() if rng is None else rng
        return np.array(
            [
                rng.uniform(*theta1_range),
                rng.uniform(*theta2_range),
                rng.uniform(*p1_range),
                rng.uniform(*p2_range),
            ],
            dtype=np.float64,
        )
