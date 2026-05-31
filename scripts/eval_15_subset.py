#!/usr/bin/env python3
"""Evaluate a single model on a small, fixed subset of eval_tasks.

- 5 shortest_path tasks
- 5 diameter tasks
- 5 components tasks

Reads from eval_tasks_metric.jsonl (or a custom path) and compares:
- Tool-based approach (with graph traversal tools)
- Raw LLM approach (with graph structure in prompt)

Prints per-task and overall metrics for both approaches.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from datasets import load_dataset

# Make sure we can import the src package when run as a script
import sys
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval_multi_model import evaluate_model_with_tools, evaluate_model_raw  # type: ignore
from src.models.model_registry import AVAILABLE_MODELS  # type: ignore


def pick_samples(data, per_task: int = 5) -> List[Dict]:
    """Pick up to `per_task` samples for each task type.

    Task types are assumed to be: shortest_path, diameter, components.
    """
    buckets: Dict[str, List[Dict]] = defaultdict(list)
    for sample in data:
        task = sample["task"]
        if len(buckets[task]) < per_task:
            buckets[task].append(sample)
        # Stop early if we've filled all three buckets
        if (
            len(buckets.get("shortest_path", [])) >= per_task
            and len(buckets.get("diameter", [])) >= per_task
            and len(buckets.get("components", [])) >= per_task
        ):
            break

    selected: List[Dict] = []
    for task in ["shortest_path", "diameter", "components"]:
        if buckets[task]:
            selected.extend(buckets[task])
    return selected


def evaluate_subset(tasks_path: Path, model_key: str, per_task: int = 5, max_steps: int = 6) -> None:
    # Load dataset
    data = load_dataset("json", data_files=str(tasks_path))["train"]

    # Pick subset
    samples = pick_samples(data, per_task=per_task)
    if not samples:
        print("No samples found in dataset.")
        return

    print(f"Evaluating model '{model_key}' on {len(samples)} samples ")
    print(f"({per_task} each of shortest_path, diameter, components where available)")
    print(f"Dataset: {tasks_path}")
    print(f"Comparing: Tool-based vs Raw LLM")
    print()

    # Aggregated metrics for both approaches
    agg_tool = {
        "overall": {"total": 0, "correct": 0},
        "shortest_path": {"total": 0, "correct": 0},
        "diameter": {"total": 0, "correct": 0},
        "components": {"total": 0, "correct": 0},
    }
    agg_raw = {
        "overall": {"total": 0, "correct": 0},
        "shortest_path": {"total": 0, "correct": 0},
        "diameter": {"total": 0, "correct": 0},
        "components": {"total": 0, "correct": 0},
    }

    for idx, sample in enumerate(samples, 1):
        task = sample["task"]
        print(f"Sample {idx}/{len(samples)}: {task} on {sample['graph_id']}")
        print(f"  Prompt: {sample['prompt']}")

        # Evaluate tool-based
        tool_result = None
        tool_exact = False
        try:
            tool_result = evaluate_model_with_tools(sample, model_key, max_steps=max_steps)
            tool_metrics = tool_result.get("metrics", {})
            tool_exact = bool(tool_metrics.get("exact_match", False))
            
            agg_tool["overall"]["total"] += 1
            agg_tool[task]["total"] += 1
            if tool_exact:
                agg_tool["overall"]["correct"] += 1
                agg_tool[task]["correct"] += 1
        except Exception as e:  # pragma: no cover - debug output
            print(f"  ❌ Tool-based error: {e}")
            agg_tool["overall"]["total"] += 1
            agg_tool[task]["total"] += 1

        # Evaluate raw LLM
        raw_result = None
        raw_exact = False
        try:
            raw_result = evaluate_model_raw(sample, model_key)
            raw_metrics = raw_result.get("metrics", {})
            raw_exact = bool(raw_metrics.get("exact_match", False))
            
            agg_raw["overall"]["total"] += 1
            agg_raw[task]["total"] += 1
            if raw_exact:
                agg_raw["overall"]["correct"] += 1
                agg_raw[task]["correct"] += 1
        except Exception as e:  # pragma: no cover - debug output
            print(f"  ❌ Raw LLM error: {e}")
            agg_raw["overall"]["total"] += 1
            agg_raw[task]["total"] += 1

        # Print comparison
        tool_status = "✓" if tool_exact else "✗"
        raw_status = "✓" if raw_exact else "✗"
        print(f"  Tool-based: {tool_status} | Raw LLM: {raw_status}")
        
        # Show detailed metrics if available
        if tool_result and "metrics" in tool_result:
            metrics = tool_result["metrics"]
            if "pred_length" in metrics:
                print(f"    Tool: length={metrics.get('pred_length')}, error={metrics.get('abs_error', 'N/A')}")
            elif "pred_diameter" in metrics:
                print(f"    Tool: diameter={metrics.get('pred_diameter')}, error={metrics.get('abs_error', 'N/A')}")
            elif "pred_count" in metrics:
                print(f"    Tool: count={metrics.get('pred_count')}, error={metrics.get('abs_error', 'N/A')}")
        
        if raw_result and "metrics" in raw_result:
            metrics = raw_result["metrics"]
            if "pred_length" in metrics:
                print(f"    Raw:  length={metrics.get('pred_length')}, error={metrics.get('abs_error', 'N/A')}")
            elif "pred_diameter" in metrics:
                print(f"    Raw:  diameter={metrics.get('pred_diameter')}, error={metrics.get('abs_error', 'N/A')}")
            elif "pred_count" in metrics:
                print(f"    Raw:  count={metrics.get('pred_count')}, error={metrics.get('abs_error', 'N/A')}")
        
        print()

    # Summary
    print("\n" + "=" * 70)
    print("Summary Metrics: Tool-based vs Raw LLM")
    print("=" * 70)

    def fmt(acc: Dict[str, int]) -> str:
        if acc["total"] == 0:
            return "N/A (0 samples)"
        pct = acc['correct'] / acc['total'] * 100
        return f"{acc['correct']}/{acc['total']} = {pct:5.1f}%"

    print(f"\n{'Task':<20} {'Tool-based':<20} {'Raw LLM':<20} {'Winner':<10}")
    print("-" * 70)
    
    for task_name in ["overall", "shortest_path", "diameter", "components"]:
        tool_acc = fmt(agg_tool[task_name])
        raw_acc = fmt(agg_raw[task_name])
        
        # Determine winner
        tool_pct = agg_tool[task_name]["correct"] / agg_tool[task_name]["total"] * 100 if agg_tool[task_name]["total"] > 0 else 0
        raw_pct = agg_raw[task_name]["correct"] / agg_raw[task_name]["total"] * 100 if agg_raw[task_name]["total"] > 0 else 0
        
        if tool_pct > raw_pct:
            winner = "Tool"
        elif raw_pct > tool_pct:
            winner = "Raw LLM"
        else:
            winner = "Tie"
        
        display_name = task_name.replace("_", " ").title()
        print(f"{display_name:<20} {tool_acc:<20} {raw_acc:<20} {winner:<10}")
    
    print("\n" + "=" * 70)


def cli() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a single model on a 15-sample subset of eval_tasks_metric.jsonl")
    parser.add_argument(
        "--tasks",
        type=Path,
        default=Path("data/eval_tasks_metric.jsonl"),
        help="Path to evaluation dataset (JSONL)",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model key (must be in AVAILABLE_MODELS)",
    )
    parser.add_argument(
        "--per-task",
        type=int,
        default=5,
        help="Number of samples per task type (default: 5)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=6,
        help="Maximum steps for tool-based agent (default: 6)",
    )

    args = parser.parse_args()

    if args.model not in AVAILABLE_MODELS:
        print("Unknown model key. Available models:")
        for key, spec in AVAILABLE_MODELS.items():
            print(f"  {key:20} - {spec.name:20} ({spec.size})")
        raise SystemExit(1)

    evaluate_subset(args.tasks, args.model, per_task=args.per_task, max_steps=args.max_steps)


if __name__ == "__main__":
    cli()
