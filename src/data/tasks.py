from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import networkx as nx

from .graphs import GRAPH_SPECS, GraphSpec, load_graph


@dataclass
class TaskSample:
    graph_id: str
    task: str  # "shortest_path" | "diameter" | "components"
    source: Optional[int] = None
    target: Optional[int] = None
    prompt: str = ""
    ground_truth: Dict[str, object] = None  # path/diameter/component_count


def _random_node_pair(graph: nx.Graph, rng: random.Random) -> Tuple[int, int]:
    nodes = list(graph.nodes())
    if len(nodes) < 2:
        raise ValueError("graph too small for node pair sampling")
    a = rng.choice(nodes)
    b = rng.choice(nodes)
    while b == a:
        b = rng.choice(nodes)
    return a, b


def generate_tasks_for_graph(
    graph_id: str,
    task_types: Iterable[str],
    num_samples: int,
    seed: int = 0,
) -> List[TaskSample]:
    """Generate synthetic task instances with ground truth for a given graph."""
    if graph_id not in GRAPH_SPECS:
        raise ValueError(f"graph_id {graph_id} not in GRAPH_SPECS")
    spec: GraphSpec = GRAPH_SPECS[graph_id]
    graph = load_graph(graph_id)
    rng = random.Random(seed)
    task_types = list(task_types)
    if not task_types:
        raise ValueError("task_types must be non-empty")

    samples: List[TaskSample] = []
    for i in range(num_samples):
        task = task_types[i % len(task_types)]
        if task == "shortest_path":
            src, dst = _random_node_pair(graph, rng)
            try:
                path = nx.shortest_path(graph, src, dst)
                gt = {"path": path, "length": len(path) - 1}
                prompt = f"Find the shortest path between node {src} and node {dst} in graph '{graph_id}'."
            except nx.NetworkXNoPath:
                gt = {"path": None, "length": None, "error": "no_path"}
                prompt = f"Attempt to find a path between node {src} and node {dst} in graph '{graph_id}'."
            samples.append(
                TaskSample(
                    graph_id=graph_id,
                    task=task,
                    source=src,
                    target=dst,
                    prompt=prompt,
                    ground_truth=gt,
                )
            )
        elif task == "diameter":
            if nx.is_connected(graph):
                diameter = nx.diameter(graph)
                note = None
            else:
                largest_cc = max(nx.connected_components(graph), key=len)
                subgraph = graph.subgraph(largest_cc)
                diameter = nx.diameter(subgraph)
                note = "computed on largest connected component"
            prompt = f"Estimate the diameter of graph '{graph_id}'."
            gt = {"diameter": diameter, "note": note}
            samples.append(TaskSample(graph_id=graph_id, task=task, prompt=prompt, ground_truth=gt))
        elif task == "components":
            comp_list = [list(c) for c in nx.connected_components(graph)]
            prompt = f"Report the connected components of graph '{graph_id}'."
            gt = {"component_count": len(comp_list)}
            samples.append(TaskSample(graph_id=graph_id, task=task, prompt=prompt, ground_truth=gt))
        else:
            raise ValueError(f"unsupported task type {task}")
    return samples


def save_tasks_jsonl(samples: List[TaskSample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(asdict(sample)) + "\n")


def load_tasks_jsonl(path: Path) -> List[TaskSample]:
    loaded: List[TaskSample] = []
    with path.open() as f:
        for line in f:
            obj = json.loads(line)
            loaded.append(
                TaskSample(
                    graph_id=obj["graph_id"],
                    task=obj["task"],
                    source=obj.get("source"),
                    target=obj.get("target"),
                    prompt=obj.get("prompt", ""),
                    ground_truth=obj.get("ground_truth"),
                )
            )
    return loaded

