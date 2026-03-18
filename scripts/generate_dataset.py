from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.dynamics import DoublePendulumHamiltonian, rollout_rk4
from src.utils import save_dataset_npz, set_seed


def main() -> None:
    set_seed(42)

    system = DoublePendulumHamiltonian(m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=9.81)
    rng = np.random.default_rng(42)

    n_trajectories = 40
    t_final = 10.0
    dt = 0.01
    t = np.arange(0.0, t_final + dt, dt)

    states = []
    derivatives = []
    energies = []

    for idx in range(n_trajectories):
        z0 = system.sample_initial_state(rng=rng)
        traj = rollout_rk4(system.vector_field_numpy, z0, t)
        z_dot = np.stack([system.vector_field_numpy(z) for z in traj], axis=0)
        H = system.energy(traj)

        states.append(traj)
        derivatives.append(z_dot)
        energies.append(H)

        if (idx + 1) % 10 == 0:
            print(f"generated {idx + 1}/{n_trajectories} trajectories")

    states = np.concatenate(states, axis=0)
    derivatives = np.concatenate(derivatives, axis=0)
    energies = np.concatenate(energies, axis=0)

    out_path = ROOT / "data" / "double_pendulum_dataset.npz"
    save_dataset_npz(out_path, states=states, derivatives=derivatives, energies=energies, t=t)

    print(f"saved dataset to {out_path}")
    print(f"states shape      : {states.shape}")
    print(f"derivatives shape : {derivatives.shape}")
    print(f"energies shape    : {energies.shape}")


if __name__ == "__main__":
    main()
