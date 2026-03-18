from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.utils import plot_aggregate_metric_bars


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an aggregate comparison bar chart from metrics_summary.json")
    parser.add_argument(
        "--metrics-json",
        type=str,
        default=str(ROOT / "artifacts" / "evaluation" / "metrics_summary.json"),
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(ROOT / "artifacts" / "evaluation" / "aggregate_results_bar.png"),
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Aggregate model comparison",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_path = Path(args.metrics_json)
    summary = json.loads(metrics_path.read_text())

    if "aggregate" not in summary:
        raise ValueError(f"No aggregate section found in {metrics_path}")

    aggregate = summary["aggregate"]
    plot_aggregate_metric_bars(aggregate, args.out, title=args.title)
    print(f"saved plot to: {args.out}")


if __name__ == "__main__":
    main()
