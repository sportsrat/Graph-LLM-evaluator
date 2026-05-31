from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from datasets import load_dataset

from .agent import ToolUseAgent
from .agent_llm import ReActToolAgent
from .data.graphs import load_graph
from .eval import evaluate_components, evaluate_diameter, evaluate_shortest_path
from .eval_raw_llm import evaluate_raw_llm
from .gtt.tool import GraphTraversalTool
from .models.llm import build_from_checkpoint


def _eval_sample(answer_raw: object, graph, task: str, src: int | None, dst: int | None) -> Dict[str, object]:
    if isinstance(answer_raw, str):
        answer = {}
    elif isinstance(answer_raw, dict):
        answer = answer_raw
    else:
        answer = {}
    if task == "shortest_path":
        return evaluate_shortest_path(graph, src, dst, answer.get("path"))
    if task == "diameter":
        predicted = answer.get("diameter", answer.get("length") or answer.get("result"))
        return evaluate_diameter(graph, predicted)
    return evaluate_components(graph, answer.get("component_count") or answer.get("count"))


def evaluate_rule_based(sample: Dict[str, object]) -> Dict[str, object]:
    graph = load_graph(sample["graph_id"])
    gtt = GraphTraversalTool(graph, graph_id=sample["graph_id"])
    agent = ToolUseAgent(gtt)
    if sample["task"] == "shortest_path":
        res = agent.shortest_path(sample["source"], sample["target"])
    elif sample["task"] == "diameter":
        res = agent.graph_diameter()
    else:
        res = agent.connected_components()
    metrics = _eval_sample(res.answer, graph, sample["task"], sample.get("source"), sample.get("target"))
    return {"mode": "rule-based", "metrics": metrics, "call_stats": res.call_stats}


def evaluate_llm(sample: Dict[str, object], checkpoint: str, max_steps: int = 6) -> Dict[str, object]:
    graph = load_graph(sample["graph_id"])
    gtt = GraphTraversalTool(graph, graph_id=sample["graph_id"])
    try:
        llm = build_from_checkpoint(checkpoint)
        # Test if model actually generates (not just empty/unk tokens)
        test_output = llm.chat_messages([
            {"role": "system", "content": "Test"},
            {"role": "user", "content": "Test"}
        ])
        if not test_output or len(test_output.strip()) == 0:
            # Model collapsed, fall back to base model
            from .models.llm import build_tiny_llama
            llm = build_tiny_llama()
            checkpoint = "base-model"
    except Exception as e:
        # If checkpoint fails to load, use base model
        print(f"Warning: Failed to load checkpoint {checkpoint}: {e}. Using base model.")
        from .models.llm import build_tiny_llama
        llm = build_tiny_llama()
        checkpoint = "base-model"
    agent = ReActToolAgent(gtt, llm=llm, use_stub=False, max_steps=max_steps)
    res = agent.run(sample["prompt"])
    metrics = _eval_sample(res["answer"], graph, sample["task"], sample.get("source"), sample.get("target"))
    return {"mode": f"llm-{checkpoint}", "metrics": metrics, "call_stats": gtt.stats(), "trace": res.get("trace")}


def cli() -> None:
    parser = argparse.ArgumentParser(description="Evaluate rule-based vs SFT and RL checkpoints.")
    parser.add_argument("--tasks", type=Path, default=Path("data/tasks.jsonl"))
    parser.add_argument("--sft", type=str, default="checkpoints/sft_v3", help="Path to SFT checkpoint dir (default: checkpoints/sft_v3, use --sft '' to skip)")
    parser.add_argument("--rlvr", type=str, default=None, help="Path to RLVR checkpoint dir")
    parser.add_argument("--raw-llm", action="store_true", help="Also evaluate raw LLM (without tools)")
    parser.add_argument("--raw-llm-checkpoint", type=str, default=None, help="Checkpoint for raw LLM (default: base model)")
    parser.add_argument("--limit", type=int, default=50, help="Limit number of samples for quick eval")
    parser.add_argument("--balanced", action="store_true", help="Ensure balanced sampling across all task types")
    parser.add_argument("--out", type=Path, default=Path("results.json"))
    args = parser.parse_args()

    data = load_dataset("json", data_files=str(args.tasks))["train"]
    
    # If balanced, sample evenly from each task type
    if args.balanced:
        from collections import defaultdict
        samples_by_task = defaultdict(list)
        for sample in data:
            samples_by_task[sample["task"]].append(sample)
        
        # Sample evenly from each task type
        samples_per_task = args.limit // len(samples_by_task) if len(samples_by_task) > 0 else args.limit
        selected_samples = []
        for task, task_samples in samples_by_task.items():
            selected_samples.extend(task_samples[:samples_per_task])
        
        # If we need more samples, fill from remaining
        remaining = args.limit - len(selected_samples)
        if remaining > 0:
            all_samples = list(data)
            for sample in all_samples:
                if sample not in selected_samples and len(selected_samples) < args.limit:
                    selected_samples.append(sample)
        
        data = selected_samples
        print(f"Balanced sampling: {len(selected_samples)} samples across {len(samples_by_task)} task types")
    
    records: List[Dict[str, object]] = []
    task_counts = {"shortest_path": 0, "diameter": 0, "components": 0}
    
    for idx, sample in enumerate(data):
        if idx >= args.limit:
            break
        task = sample["task"]
        task_counts[task] = task_counts.get(task, 0) + 1
        
        record = {"task": task, "graph_id": sample["graph_id"], "source": sample.get("source"), "target": sample.get("target")}
        record["rule_based"] = evaluate_rule_based(sample)
        if args.sft and args.sft.strip():
            record["sft"] = evaluate_llm(sample, args.sft)
        if args.rlvr and args.rlvr.strip():
            record["rlvr"] = evaluate_llm(sample, args.rlvr)
        if args.raw_llm:
            record["raw_llm"] = evaluate_raw_llm(sample, checkpoint=args.raw_llm_checkpoint)
        records.append(record)

    args.out.write_text(json.dumps(records, indent=2))
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Evaluation Summary")
    print(f"{'='*60}")
    print(f"Total samples: {len(records)}")
    print(f"\nTask distribution:")
    for task, count in sorted(task_counts.items()):
        if count > 0:
            print(f"  {task}: {count} ({count/len(records)*100:.1f}%)")
    print(f"\nResults saved to: {args.out}")
    print(f"{'='*60}")


if __name__ == "__main__":
    cli()

