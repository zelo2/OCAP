from __future__ import annotations

import argparse
from statistics import mean
import json
from pathlib import Path
import sys
from typing import Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ocap.config import load_config
from ocap.data import load_stream
from ocap.framework import OCAP


def run(
    dataset: str,
    data: str | Path,
    output: str | Path,
    seeds: Sequence[int],
    device: str,
    torch_threads: int,
) -> None:
    config = load_config(dataset=dataset)
    config.device = device
    stream, feature_names = load_stream(data)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, int(torch_threads)))
    seed_results = []
    for seed in seeds:
        config.seed = int(seed)
        ocap = OCAP(len(feature_names), config)
        metrics = ocap.run(stream)
        run_dir = root / f"{dataset}_main_seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
        print(f"{dataset} seed {seed}", json.dumps(metrics))
        seed_results.append(metrics)

    if seed_results:
        averages = {}
        for name in seed_results[0]:
            values = [row[name] for row in seed_results]
            averages[name] = mean(values) if all(value is not None for value in values) else None
        print(f"{dataset} mean", json.dumps(averages))


def _csv_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the fixed OCAP main-experiment configuration",
    )
    parser.add_argument("--dataset", required=True, choices=("ces", "globem"))
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args(argv)
    run(
        args.dataset, args.data, args.output,
        _csv_ints(args.seeds), args.device, args.torch_threads,
    )


if __name__ == "__main__":
    main()
