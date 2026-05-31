from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import networkx as nx
from src.data.graphs import GRAPH_SPECS, load_graph  # noqa: E402
from src.data.tasks import TaskSample, save_tasks_jsonl  # noqa: E402


def generate_eval_tasks_for_graph(
    graph_id: str,
    num_shortest_path: int = 100,
    num_diameter: int = 5,
    num_components: int = 5,
    seed: int = 9999,
) -> list[TaskSample]:
    """
    Generate a diverse evaluation dataset for a large graph.
    
    This function creates a balanced mix of tasks with strategic sampling:
    - Shortest path: Mix of easy (direct neighbors), medium, and long paths
    - Diameter: Always included (deterministic)
    - Components: Always included (deterministic)
    """
    import random
    
    if graph_id not in GRAPH_SPECS:
        raise ValueError(f"graph_id {graph_id} not in GRAPH_SPECS")
    
    print(f"  Loading graph {graph_id}...")
    graph = load_graph(graph_id)
    print(f"  Graph loaded: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    rng = random.Random(seed)
    samples: list[TaskSample] = []
    
    # Generate shortest path tasks with diversity
    nodes = list(graph.nodes())
    if len(nodes) >= 2:
        print(f"  Generating {num_shortest_path} shortest path tasks...")
        # Sample diverse shortest path tasks
        path_samples = []
        seen_pairs = set()  # Track unique (src, dst) pairs
        attempts = 0
        max_attempts = num_shortest_path * 30  # More attempts for large graphs
        
        while len(path_samples) < num_shortest_path and attempts < max_attempts:
            attempts += 1
            if attempts % 1000 == 0:
                print(f"    Attempt {attempts}/{max_attempts}, found {len(path_samples)}/{num_shortest_path} paths")
            
            src, dst = rng.sample(nodes, 2)
            
            # Skip if we've seen this pair before
            pair_key = (min(src, dst), max(src, dst))
            if pair_key in seen_pairs:
                continue
            
            seen_pairs.add(pair_key)
            
            try:
                path = nx.shortest_path(graph, src, dst)
                path_length = len(path) - 1
                
                # Ensure diversity: include mix of path lengths
                path_samples.append((src, dst, path, path_length))
            except nx.NetworkXNoPath:
                # Include some disconnected node pairs to test error handling
                disconnected_count = len([p for p in path_samples if p[3] is None])
                if disconnected_count < num_shortest_path * 0.05:  # ~5% disconnected
                    path_samples.append((src, dst, None, None))
        
        # Sort by path length for diversity, then sample evenly
        path_samples.sort(key=lambda x: x[3] if x[3] is not None else 999)
        
        # Sample evenly across path lengths to ensure diversity
        if len(path_samples) > num_shortest_path:
            step = max(1, len(path_samples) // num_shortest_path)
            selected_paths = path_samples[::step][:num_shortest_path]
        else:
            selected_paths = path_samples
        
        print(f"  Creating {len(selected_paths)} shortest path task samples...")
        for src, dst, path, length in selected_paths:
            if path is None:
                gt = {"path": None, "length": None, "error": "no_path"}
                prompt = f"Attempt to find a path between node {src} and node {dst} in graph '{graph_id}'."
            else:
                gt = {"path": path, "length": length}
                prompt = f"Find the shortest path between node {src} and node {dst} in graph '{graph_id}'."
            
            samples.append(
                TaskSample(
                    graph_id=graph_id,
                    task="shortest_path",
                    source=src,
                    target=dst,
                    prompt=prompt,
                    ground_truth=gt,
                )
            )
    
    # Generate diameter tasks (deterministic, but include multiple for consistency)
    print(f"  Generating {num_diameter} diameter tasks...")
    for _ in range(num_diameter):
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
        samples.append(
            TaskSample(
                graph_id=graph_id,
                task="diameter",
                prompt=prompt,
                ground_truth=gt,
            )
        )
    
    # Generate components tasks (deterministic)
    print(f"  Generating {num_components} components tasks...")
    for _ in range(num_components):
        comp_list = [list(c) for c in nx.connected_components(graph)]
        prompt = f"Report the connected components of graph '{graph_id}'."
        gt = {"component_count": len(comp_list)}
        samples.append(
            TaskSample(
                graph_id=graph_id,
                task="components",
                prompt=prompt,
                ground_truth=gt,
            )
        )
    
    return samples


def cli() -> None:
    parser = argparse.ArgumentParser(
        description="Generate evaluation dataset for large graph reasoning tasks."
    )
    parser.add_argument(
        "--graphs",
        nargs="+",
        default=["gnp-2k", "gnp-5k", "gnp-10k", "gnp-20k"],
        help="Graph ids to include (default: large GNP graphs).",
    )
    parser.add_argument(
        "--shortest-path-per-graph",
        type=int,
        default=100,
        help="Number of shortest path tasks per graph (default: 100).",
    )
    parser.add_argument(
        "--diameter-per-graph",
        type=int,
        default=5,
        help="Number of diameter tasks per graph (default: 5).",
    )
    parser.add_argument(
        "--components-per-graph",
        type=int,
        default=5,
        help="Number of components tasks per graph (default: 5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=9999,
        help="Random seed for reproducibility (default: 9999, different from training).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/eval_tasks_large.jsonl"),
        help="Output path for evaluation dataset (default: data/eval_tasks_large.jsonl).",
    )
    args = parser.parse_args()

    all_samples = []
    current_seed = args.seed
    
    for gid in args.graphs:
        print(f"\n{'='*60}")
        print(f"Generating evaluation tasks for {gid}...")
        print(f"{'='*60}")
        try:
            samples = generate_eval_tasks_for_graph(
                gid,
                num_shortest_path=args.shortest_path_per_graph,
                num_diameter=args.diameter_per_graph,
                num_components=args.components_per_graph,
                seed=current_seed,
            )
            all_samples.extend(samples)
            current_seed += 1  # Vary seed per graph for diversity
            print(f"  ✓ Generated {len(samples)} tasks for {gid}")
        except Exception as e:
            print(f"  ✗ Error generating tasks for {gid}: {e}")
            continue
    
    save_tasks_jsonl(all_samples, args.out)
    
    # Print summary statistics
    task_counts = {}
    graph_counts = {}
    for sample in all_samples:
        task_counts[sample.task] = task_counts.get(sample.task, 0) + 1
        graph_counts[sample.graph_id] = graph_counts.get(sample.graph_id, 0) + 1
    
    print(f"\n{'='*60}")
    print(f"Large Graph Evaluation Dataset Summary")
    print(f"{'='*60}")
    print(f"Total samples: {len(all_samples)}")
    print(f"\nTask distribution:")
    for task, count in sorted(task_counts.items()):
        print(f"  {task}: {count}")
    print(f"\nGraph distribution:")
    for graph_id, count in sorted(graph_counts.items()):
        print(f"  {graph_id}: {count}")
    print(f"\nSaved to: {args.out}")
    print(f"{'='*60}")


if __name__ == "__main__":
    cli()

