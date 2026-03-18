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


def plot_aggregate_metric_bars(
    metrics: dict[str, float],
    save_path: str | Path,
    title: str = "Aggregate model comparison",
) -> None:
    """
    Plot a simple grouped bar chart for the two most important aggregate metrics:
    trajectory MSE and maximum energy drift.

    Expected keys in ``metrics``:
        - mean_baseline_mse
        - mean_hnn_mse
        - mean_baseline_max_energy_drift
        - mean_hnn_max_energy_drift
    """
    save_path = _prepare_path(save_path)

    labels = ["Trajectory MSE", "Max energy drift"]
    baseline_vals = [
        float(metrics["mean_baseline_mse"]),
        float(metrics["mean_baseline_max_energy_drift"]),
    ]
    hnn_vals = [
        float(metrics["mean_hnn_mse"]),
        float(metrics["mean_hnn_max_energy_drift"]),
    ]

    x = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, baseline_vals, width, label="baseline")
    bars2 = ax.bar(x + width / 2, hnn_vals, width, label="hnn")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("metric value")
    ax.set_title(title)
    ax.legend()
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.25)

    def _annotate(bars, vals):
        for bar, val in zip(bars, vals):
            ax.annotate(
                f"{val:.3g}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    _annotate(bars1, baseline_vals)
    _annotate(bars2, hnn_vals)

    if hnn_vals[0] > 0 and hnn_vals[1] > 0:
        mse_ratio = baseline_vals[0] / hnn_vals[0]
        drift_ratio = baseline_vals[1] / hnn_vals[1]
        ratio_text = (
            f"MSE improvement: {mse_ratio:.1f}x\n"
            f"Drift improvement: {drift_ratio:.1f}x"
        )
        ax.text(
            0.98,
            0.98,
            ratio_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(save_path, dpi=220)
    plt.close(fig)
