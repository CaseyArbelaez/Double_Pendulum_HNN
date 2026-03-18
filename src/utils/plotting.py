from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


STATE_LABELS = [r"$\theta_1$", r"$\theta_2$", r"$p_1$", r"$p_2$"]


def _prepare_path(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_training_curves(history: dict, save_path: str | Path, title: str) -> None:
    save_path = _prepare_path(save_path)
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, history["train_loss"], label="train")
    if "val_loss" in history and len(history["val_loss"]) == len(epochs):
        plt.plot(epochs, history["val_loss"], label="val")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("MSE loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()



def plot_state_rollout(
    t: np.ndarray,
    true_traj: np.ndarray,
    pred_trajs: dict[str, np.ndarray],
    save_path: str | Path,
    title: str,
) -> None:
    save_path = _prepare_path(save_path)
    fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)

    for i, ax in enumerate(axes):
        ax.plot(t, true_traj[:, i], label="true", linewidth=2.0)
        for name, traj in pred_trajs.items():
            ax.plot(t, traj[:, i], label=name, alpha=0.9)
        ax.set_ylabel(STATE_LABELS[i])
        ax.grid(alpha=0.25)

    axes[0].set_title(title)
    axes[-1].set_xlabel("time")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)



def plot_energy_drift(
    t: np.ndarray,
    true_energy: np.ndarray,
    pred_energies: dict[str, np.ndarray],
    save_path: str | Path,
    title: str,
) -> None:
    save_path = _prepare_path(save_path)
    baseline_energy = float(true_energy[0])

    plt.figure(figsize=(8, 4.5))
    plt.plot(t, np.abs(true_energy - baseline_energy), label="true", linewidth=2.0)
    for name, energy in pred_energies.items():
        plt.plot(t, np.abs(energy - baseline_energy), label=name)
    plt.yscale("log")
    plt.xlabel("time")
    plt.ylabel(r"$|H(t) - H(0)|$")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()



def plot_phase_portrait(
    trajectories: dict[str, np.ndarray],
    save_path: str | Path,
    title: str,
    dims: tuple[int, int] = (0, 2),
) -> None:
    save_path = _prepare_path(save_path)
    i, j = dims

    plt.figure(figsize=(6.5, 5.5))
    for name, traj in trajectories.items():
        plt.plot(traj[:, i], traj[:, j], label=name, linewidth=1.3)
    plt.xlabel(STATE_LABELS[i])
    plt.ylabel(STATE_LABELS[j])
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()



def plot_dataset_examples(
    t: np.ndarray,
    trajectories: Iterable[np.ndarray],
    save_path: str | Path,
    title: str = "Sample training trajectories",
) -> None:
    save_path = _prepare_path(save_path)
    trajectories = list(trajectories)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    axes = axes.ravel()
    for idx, traj in enumerate(trajectories[: len(axes)]):
        axes[idx].plot(t, traj[:, 0], label=r"$\theta_1$")
        axes[idx].plot(t, traj[:, 1], label=r"$\theta_2$")
        axes[idx].set_title(f"trajectory {idx + 1}")
        axes[idx].grid(alpha=0.25)
    axes[0].legend()
    for ax in axes[-2:]:
        ax.set_xlabel("time")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
