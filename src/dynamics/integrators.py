from __future__ import annotations

from typing import Callable

import numpy as np


Array = np.ndarray
VectorField = Callable[[Array], Array]


def rk4_step(f: VectorField, z: Array, dt: float) -> Array:
    """One RK4 step for z' = f(z)."""
    k1 = f(z)
    k2 = f(z + 0.5 * dt * k1)
    k3 = f(z + 0.5 * dt * k2)
    k4 = f(z + dt * k3)
    return z + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rollout_rk4(f: VectorField, z0: Array, t: Array) -> Array:
    """
    Roll out a trajectory over time grid t.

    Parameters
    ----------
    f : callable
        Vector field z' = f(z)
    z0 : np.ndarray
        Shape (d,)
    t : np.ndarray
        Shape (n_steps,)

    Returns
    -------
    np.ndarray
        Shape (n_steps, d)
    """
    traj = np.zeros((len(t), len(z0)), dtype=np.float64)
    traj[0] = z0
    for i in range(len(t) - 1):
        dt = t[i + 1] - t[i]
        traj[i + 1] = rk4_step(f, traj[i], float(dt))
    return traj
