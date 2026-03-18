from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset, random_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.models import BaselineVectorFieldNN
from src.training import baseline_vector_field_loss
from src.utils import load_dataset_npz, plot_training_curves, save_dataset_npz, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline vector-field NN on double pendulum data.")
    parser.add_argument("--data", type=str, default=str(ROOT / "data" / "double_pendulum_dataset.npz"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--outdir", type=str, default=str(ROOT / "artifacts" / "baseline"))
    return parser.parse_args()


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            loss = baseline_vector_field_loss(model, xb, yb)
            total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data = load_dataset_npz(args.data)
    x = torch.tensor(data["states"], dtype=torch.float32)
    y = torch.tensor(data["derivatives"], dtype=torch.float32)

    dataset = TensorDataset(x, y)
    n_val = int(len(dataset) * args.val_fraction)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = BaselineVectorFieldNN(hidden_dim=args.hidden_dim, depth=args.depth).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            loss = baseline_vector_field_loss(model, xb, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        val_loss = evaluate(model, val_loader, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), outdir / "best_model.pt")

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(
                f"epoch {epoch:03d} | train = {train_loss:.6e} | val = {val_loss:.6e}"
            )

    torch.save(model.state_dict(), outdir / "last_model.pt")
    save_dataset_npz(outdir / "history.npz", **{k: np.array(v) for k, v in history.items()})
    plot_training_curves(history, outdir / "loss_curve.png", "Baseline NN training")

    config = {
        "model": "baseline",
        "data": str(args.data),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "depth": args.depth,
        "lr": args.lr,
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "best_val_loss": best_val,
    }
    (outdir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"saved artifacts to {outdir}")


if __name__ == "__main__":
    main()
