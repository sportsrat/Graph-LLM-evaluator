from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.data.graphs import GRAPH_SPECS  # noqa: E402
from src.data.tasks import generate_tasks_for_graph, save_tasks_jsonl  # noqa: E402


def cli() -> None:
    parser = argparse.ArgumentParser(description="Generate task JSONL for graph reasoning tasks.")
    parser.add_argument("--graphs", nargs="+", default=list(GRAPH_SPECS.keys()), help="Graph ids to include.")
    parser.add_argument("--tasks", nargs="+", default=["shortest_path", "diameter", "components"])
    parser.add_argument("--per-graph", type=int, default=200, help="Tasks per graph.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("data/tasks.jsonl"))
    args = parser.parse_args()

    all_samples = []
    for gid in args.graphs:
        samples = generate_tasks_for_graph(gid, args.tasks, args.per_graph, seed=args.seed)
        all_samples.extend(samples)
        args.seed += 1  # vary seed per graph for diversity
    save_tasks_jsonl(all_samples, args.out)
    print(f"Wrote {len(all_samples)} samples to {args.out}")


if __name__ == "__main__":
    cli()

