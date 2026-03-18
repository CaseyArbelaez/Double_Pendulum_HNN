from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.dynamics import DoublePendulumHamiltonian, rollout_rk4
from src.models import BaselineVectorFieldNN, HamiltonianNN
from src.utils import (
    plot_energy_drift,
    plot_phase_portrait,
    plot_state_rollout,
    save_dataset_npz,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained NN and HNN models by rollout.")
    parser.add_argument("--baseline-ckpt", type=str, default=str(ROOT / "artifacts" / "baseline" / "best_model.pt"))
    parser.add_argument("--hnn-ckpt", type=str, default=str(ROOT / "artifacts" / "hnn" / "best_model.pt"))
    parser.add_argument("--t-final", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--n-ics", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--outdir", type=str, default=str(ROOT / "artifacts" / "evaluation"))
    return parser.parse_args()


def make_baseline_field(model: BaselineVectorFieldNN, device: torch.device):
    def field(z: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            z_t = torch.tensor(z, dtype=torch.float32, device=device).unsqueeze(0)
            out = model(z_t).squeeze(0).cpu().numpy().astype(np.float64)
        return out

    return field


def make_hnn_field(model: HamiltonianNN, device: torch.device):
    def field(z: np.ndarray) -> np.ndarray:
        z_t = torch.tensor(z, dtype=torch.float32, device=device).unsqueeze(0).requires_grad_(True)
        out = model.time_derivative(z_t).squeeze(0).detach().cpu().numpy().astype(np.float64)
        return out

    return field


def traj_metrics(true_traj: np.ndarray, pred_traj: np.ndarray, true_energy: np.ndarray, pred_energy: np.ndarray) -> dict:
    mse = float(np.mean((true_traj - pred_traj) ** 2))
    max_abs = float(np.max(np.abs(true_traj - pred_traj)))
    drift = np.abs(pred_energy - pred_energy[0])
    true_drift = np.abs(true_energy - true_energy[0])
    return {
        "trajectory_mse": mse,
        "max_abs_state_error": max_abs,
        "mean_energy_drift": float(np.mean(drift)),
        "max_energy_drift": float(np.max(drift)),
        "mean_true_energy_drift": float(np.mean(true_drift)),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    system = DoublePendulumHamiltonian(m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=9.81)
    t = np.arange(0.0, args.t_final + args.dt, args.dt)
    rng = np.random.default_rng(args.seed)

    baseline = BaselineVectorFieldNN(hidden_dim=args.hidden_dim, depth=args.depth).to(device)
    baseline.load_state_dict(torch.load(args.baseline_ckpt, map_location=device))
    baseline.eval()

    hnn = HamiltonianNN(hidden_dim=args.hidden_dim, depth=args.depth).to(device)
    hnn.load_state_dict(torch.load(args.hnn_ckpt, map_location=device))
    hnn.eval()

    baseline_field = make_baseline_field(baseline, device)
    hnn_field = make_hnn_field(hnn, device)

    summary: dict[str, dict] = {}

    for idx in range(args.n_ics):
        z0 = system.sample_initial_state(rng=rng)
        true_traj = rollout_rk4(system.vector_field_numpy, z0, t)
        baseline_traj = rollout_rk4(baseline_field, z0, t)
        hnn_traj = rollout_rk4(hnn_field, z0, t)

        true_energy = system.energy(true_traj)
        baseline_energy = system.energy(baseline_traj)
        hnn_energy = system.energy(hnn_traj)

        ic_dir = outdir / f"ic_{idx + 1:02d}"
        ic_dir.mkdir(parents=True, exist_ok=True)

        plot_state_rollout(
            t,
            true_traj,
            {"baseline": baseline_traj, "hnn": hnn_traj},
            ic_dir / "state_rollout.png",
            f"State rollout comparison (IC {idx + 1})",
        )
        plot_energy_drift(
            t,
            true_energy,
            {"baseline": baseline_energy, "hnn": hnn_energy},
            ic_dir / "energy_drift.png",
            f"Energy drift comparison (IC {idx + 1})",
        )
        plot_phase_portrait(
            {"true": true_traj, "baseline": baseline_traj, "hnn": hnn_traj},
            ic_dir / "phase_portrait_theta1_p1.png",
            f"Phase portrait (IC {idx + 1})",
            dims=(0, 2),
        )

        baseline_metrics = traj_metrics(true_traj, baseline_traj, true_energy, baseline_energy)
        hnn_metrics = traj_metrics(true_traj, hnn_traj, true_energy, hnn_energy)
        summary[f"ic_{idx + 1:02d}"] = {
            "z0": z0.tolist(),
            "baseline": baseline_metrics,
            "hnn": hnn_metrics,
        }

        save_dataset_npz(
            ic_dir / "rollouts.npz",
            t=t,
            z0=z0,
            true_traj=true_traj,
            baseline_traj=baseline_traj,
            hnn_traj=hnn_traj,
            true_energy=true_energy,
            baseline_energy=baseline_energy,
            hnn_energy=hnn_energy,
        )

        print(f"finished IC {idx + 1}/{args.n_ics}")
        print(f"  baseline mse = {baseline_metrics['trajectory_mse']:.6e}")
        print(f"  hnn mse      = {hnn_metrics['trajectory_mse']:.6e}")

    baseline_mse = [summary[k]["baseline"]["trajectory_mse"] for k in summary]
    hnn_mse = [summary[k]["hnn"]["trajectory_mse"] for k in summary]
    baseline_drift = [summary[k]["baseline"]["max_energy_drift"] for k in summary]
    hnn_drift = [summary[k]["hnn"]["max_energy_drift"] for k in summary]

    aggregate = {
        "mean_baseline_mse": float(np.mean(baseline_mse)),
        "mean_hnn_mse": float(np.mean(hnn_mse)),
        "mean_baseline_max_energy_drift": float(np.mean(baseline_drift)),
        "mean_hnn_max_energy_drift": float(np.mean(hnn_drift)),
    }
    summary["aggregate"] = aggregate
    (outdir / "metrics_summary.json").write_text(json.dumps(summary, indent=2))
    print("aggregate summary:")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
