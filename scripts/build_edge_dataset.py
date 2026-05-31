#!/usr/bin/env python3
"""Generate enriched datasets with explicit edge lists and ground-truth answers.

This script reads the existing JSONL evaluation datasets (e.g. data/eval_tasks.jsonl),
looks up the corresponding graph edge lists, and writes out new JSONL files where
each row contains:

- graph_id
- task (shortest_path / diameter / components)
- edges: full edge list of the graph as [[u, v], ...]
- source, target (for shortest_path samples; None otherwise)
- prompt (copied from the input dataset)
- answer: the ground-truth field from the original dataset

Run, for example:

    python scripts/build_edge_dataset.py \
        --in data/eval_tasks.jsonl \
        --out data/eval_tasks_with_edges.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

from src.data.graphs import load_graph


def _get_edges_for_graph(graph_id: str) -> List[Tuple[int, int]]:
    """Load graph and return its edge list as a list of (u, v) pairs."""
    graph = load_graph(graph_id)
    # Ensure we have simple (u, v) integer pairs
    return [(int(u), int(v)) for u, v in graph.edges()]


def build_edge_dataset(in_path: Path, out_path: Path) -> None:
    """Create a JSONL dataset with explicit edge lists and ground-truth answers."""
    print(f"Reading input from: {in_path}")
    print(f"Writing enriched dataset to: {out_path}")

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    # Cache edge lists per graph_id so we only load each graph once
    edge_cache: Dict[str, List[Tuple[int, int]]] = {}

    with in_path.open("r", encoding="utf-8") as fin, out_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)

            graph_id = sample["graph_id"]
            task = sample["task"]

            if graph_id not in edge_cache:
                edge_cache[graph_id] = _get_edges_for_graph(graph_id)

            record = {
                "graph_id": graph_id,
                "task": task,
                "edges": edge_cache[graph_id],
                # For shortest_path these are ints; for diameter/components they are null
                "source": sample.get("source"),
                "target": sample.get("target"),
                # Keep the original natural-language prompt for reference
                "prompt": sample.get("prompt"),
                # Ground-truth answer from the original dataset
                "answer": sample.get("ground_truth"),
            }

            fout.write(json.dumps(record) + "\n")

    print("Done.")


def cli() -> None:
    parser = argparse.ArgumentParser(
        description="Build datasets that include edge lists and ground-truth answers.",
    )
    parser.add_argument(
        "--in",
        dest="in_path",
        type=Path,
        default=Path("data/eval_tasks.jsonl"),
        help="Input JSONL file (e.g., data/eval_tasks.jsonl)",
    )
    parser.add_argument(
        "--out",
        dest="out_path",
        type=Path,
        default=Path("data/eval_tasks_with_edges.jsonl"),
        help="Output JSONL file to write enriched samples to",
    )
    args = parser.parse_args()

    build_edge_dataset(args.in_path, args.out_path)


if __name__ == "__main__":
    cli()
