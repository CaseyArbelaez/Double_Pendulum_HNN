from __future__ import annotations

from pathlib import Path

import numpy as np


def save_dataset_npz(path: str | Path, **arrays) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def load_dataset_npz(path: str | Path) -> dict:
    data = np.load(path)
    return {key: data[key] for key in data.files}
