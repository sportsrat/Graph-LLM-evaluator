"""
Multi-model evaluation pipeline.

Evaluates multiple models (Stub, TinyLlama, Qwen, Mistral, etc.) on the same dataset for comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from datasets import load_dataset

from .agent_llm import ReActToolAgent
from .data.graphs import load_graph
from .eval import evaluate_components, evaluate_diameter, evaluate_shortest_path
from .eval_raw_llm import evaluate_raw_llm
from .gtt.tool import GraphTraversalTool
from .models.llm import StubLLM, TransformersLLM
from .models.model_registry import (
    AVAILABLE_MODELS,
    build_model,
    get_cached_model,
    get_model_set,
    list_available_models,
)


def _eval_sample(answer_raw: object, graph, task: str, src: Optional[int] = None, dst: Optional[int] = None) -> Dict[str, object]:
    """Evaluate a sample answer."""
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


def evaluate_model_with_tools(
    sample: Dict,
    model_key: str,
    max_steps: int = 6,
    llm: Optional[Union[TransformersLLM, StubLLM]] = None,
) -> Dict:
    """
    Evaluate a model using the tool-based approach.
    """
    graph = load_graph(sample["graph_id"])
    gtt = GraphTraversalTool(graph, graph_id=sample["graph_id"])
    
    try:
        if llm is None:
            llm = get_cached_model(model_key)
        model_name = AVAILABLE_MODELS.get(model_key, {}).name if hasattr(AVAILABLE_MODELS.get(model_key), "name") else model_key
    except Exception as e:
        return {
            "mode": f"tool-{model_key}",
            "error": str(e),
            "metrics": {"correct": False, "error": "model_load_failed"},
        }
    
    try:
        use_stub = isinstance(llm, StubLLM)
        agent = ReActToolAgent(gtt, llm=llm, use_stub=use_stub, max_steps=max_steps)
        res = agent.run(sample["prompt"])
        metrics = _eval_sample(res["answer"], graph, sample["task"], sample.get("source"), sample.get("target"))
        return {
            "mode": f"tool-{model_key}",
            "model_name": model_name,
            "metrics": metrics,
            "call_stats": gtt.stats(),
            "trace": res.get("trace"),
        }
    except Exception as e:
        return {
            "mode": f"tool-{model_key}",
            "model_name": model_name,
            "error": str(e),
            "metrics": {"correct": False, "error": "evaluation_failed"},
        }


def evaluate_model_raw(
    sample: Dict,
    model_key: str,
    llm: Optional[Union[TransformersLLM, StubLLM]] = None,
) -> Dict:
    """
    Evaluate a model using raw LLM (no tools).
    """
    try:
        if llm is None:
            llm = get_cached_model(model_key)
        model_name = AVAILABLE_MODELS.get(model_key, {}).name if hasattr(AVAILABLE_MODELS.get(model_key), "name") else model_key
    except Exception as e:
        return {
            "mode": f"raw-{model_key}",
            "error": str(e),
            "metrics": {"correct": False, "error": "model_load_failed"},
        }
    
    from .eval_raw_llm import _build_raw_llm_prompt, _extract_answer_from_raw_llm, _eval_sample
    from .data.graphs import load_graph
    
    graph = load_graph(sample["graph_id"])
    prompt = _build_raw_llm_prompt(
        sample["prompt"],
        sample["graph_id"],
        graph,
        sample.get("source"),
        sample.get("target"),
    )
    
    messages = [
        {
            "role": "system",
            "content": "You are a graph reasoning assistant. Analyze the graph structure and provide accurate answers in JSON format."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    
    try:
        completion = llm.chat_messages(messages)
        answer = _extract_answer_from_raw_llm(completion, sample["task"])
        
        if answer is None:
            answer = {"raw_completion": completion}
        
        metrics = _eval_sample(
            answer,
            graph,
            sample["task"],
            sample.get("source"),
            sample.get("target"),
        )
        
        return {
            "mode": f"raw-{model_key}",
            "model_name": model_name,
            "metrics": metrics,
            "completion": completion,
            "answer_extracted": answer,
            "graph_size": {"nodes": graph.number_of_nodes(), "edges": graph.number_of_edges()},
        }
    except Exception as e:
        return {
            "mode": f"raw-{model_key}",
            "model_name": model_name,
            "error": str(e),
            "metrics": {"correct": False, "error": "generation_failed"},
        }


def cli() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate multiple models on graph reasoning tasks."
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        default=Path("data/eval_tasks.jsonl"),
        help="Path to evaluation dataset",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="small",
        help="Comma-separated model keys or predefined set (stub, small, medium, all-small, qwen-only, mistral-only, all). Use 'list' to see available models.",
    )
    parser.add_argument(
        "--raw-llm",
        action="store_true",
        help="Also evaluate raw LLM (without tools) for each model",
    )
    parser.add_argument(
        "--tool-based",
        action="store_true",
        default=True,
        help="Evaluate tool-based approach (default: True)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Limit number of samples for quick eval",
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="Ensure balanced sampling across all task types (shortest_path, diameter, components)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results_multi_model.json"),
        help="Output path for results",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=6,
        help="Maximum steps for tool-based agent",
    )
    args = parser.parse_args()
    
    # Handle list command
    if args.models == "list":
        print("Available models:")
        for key, spec in AVAILABLE_MODELS.items():
            print(f"  {key:20} - {spec.name:20} ({spec.size:6}) - {spec.family}")
        print("\nPredefined sets:")
        from .models.model_registry import MODEL_SETS
        for set_name, models in MODEL_SETS.items():
            print(f"  {set_name}: {', '.join(models)}")
        return
    
    # Get model list
    model_keys = get_model_set(args.models)
    print(f"Evaluating {len(model_keys)} models: {', '.join(model_keys)}")
    print(f"Dataset: {args.tasks}")
    print(f"Limit: {args.limit} samples")
    print()
    
    # Load dataset
    data = load_dataset("json", data_files=str(args.tasks))["train"]
    
    # If balanced, sample evenly from each task type
    if args.balanced:
        from collections import defaultdict
        samples_by_task = defaultdict(list)
        for sample in data:
            samples_by_task[sample["task"]].append(sample)
        
        samples_per_task = args.limit // len(samples_by_task) if len(samples_by_task) > 0 else args.limit
        selected_samples = []
        for task, task_samples in samples_by_task.items():
            selected_samples.extend(task_samples[:samples_per_task])
        
        remaining = args.limit - len(selected_samples)
        if remaining > 0:
            all_samples = list(data)
            for sample in all_samples:
                if sample not in selected_samples and len(selected_samples) < args.limit:
                    selected_samples.append(sample)
        
        data = selected_samples
        print(f"Balanced sampling: {len(selected_samples)} samples across {len(samples_by_task)} task types")
        print()
    
    records: List[Dict] = []
    task_counts = {"shortest_path": 0, "diameter": 0, "components": 0}
    
    # Pre-load/cache models to avoid reloading
    loaded_models: Dict[str, Any] = {}
    for model_key in model_keys:
        try:
            print(f"Preparing model {model_key}...", end=" ", flush=True)
            loaded_models[model_key] = get_cached_model(model_key)
            print("Done")
        except Exception as e:
            print(f"Failed to load: {e}")
            loaded_models[model_key] = None

    for idx, sample in enumerate(data):
        if idx >= args.limit:
            break
        
        task = sample["task"]
        task_counts[task] = task_counts.get(task, 0) + 1
        
        print(f"Sample {idx + 1}/{min(args.limit, len(data))}: {sample['task']} on {sample['graph_id']}")
        
        record = {
            "task": sample["task"],
            "graph_id": sample["graph_id"],
            "source": sample.get("source"),
            "target": sample.get("target"),
            "prompt": sample.get("prompt"),
        }
        
        # Evaluate each model
        for model_key in model_keys:
            llm_inst = loaded_models.get(model_key)
            print(f"  Evaluating {model_key}...", end=" ", flush=True)
            
            # Tool-based evaluation
            if args.tool_based:
                try:
                    result = evaluate_model_with_tools(sample, model_key, max_steps=args.max_steps, llm=llm_inst)
                    record[f"tool_{model_key}"] = result
                    if "error" not in result:
                        correct = result.get("metrics", {}).get("exact_match", False)
                        print(f"Tool: {'✓' if correct else '✗'}", end=" ")
                    else:
                        print(f"Tool: ✗", end=" ")
                except Exception as e:
                    print(f"Tool: ✗ Exception", end=" ")
                    record[f"tool_{model_key}"] = {"error": str(e)}
            
            # Raw LLM evaluation
            if args.raw_llm:
                try:
                    result = evaluate_model_raw(sample, model_key, llm=llm_inst)
                    record[f"raw_{model_key}"] = result
                    if "error" not in result:
                        correct = result.get("metrics", {}).get("exact_match", False)
                        print(f"Raw: {'✓' if correct else '✗'}")
                    else:
                        print(f"Raw: ✗")
                except Exception as e:
                    print(f"Raw: ✗ Exception")
                    record[f"raw_{model_key}"] = {"error": str(e)}
            
            if not args.raw_llm:
                print()
        
        records.append(record)
        print()
    
    # Save results
    args.out.write_text(json.dumps(records, indent=2))
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Evaluation Summary")
    print(f"{'='*60}")
    print(f"Total samples: {len(records)}")
    print(f"Models evaluated: {len(model_keys)}")
    print(f"\nTask distribution:")
    for task, count in sorted(task_counts.items()):
        if count > 0:
            print(f"  {task}: {count} ({count/len(records)*100:.1f}%)")
    print(f"\nResults saved to: {args.out}")
    print(f"{'='*60}")


if __name__ == "__main__":
    cli()
