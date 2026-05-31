#!/usr/bin/env python3
"""
Analyze multi-model evaluation results.

Generates comprehensive comparison reports across multiple models.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def analyze_multi_model_results(results: List[Dict]) -> Dict:
    """Analyze results from multi-model evaluation."""
    analysis = {
        "total_samples": len(results),
        "models": {},
        "by_task": defaultdict(lambda: defaultdict(lambda: {"total": 0, "correct": 0, "errors": 0})),
        "by_graph_size": defaultdict(lambda: defaultdict(lambda: {"total": 0, "correct": 0, "errors": 0})),
        "comparison": {
            "tool_based": {},
            "raw_llm": {},
        },
    }
    
    # Process each record
    for record in results:
        task = record.get("task", "unknown")
        graph_id = record.get("graph_id", "unknown")
        
        # Categorize graph size
        if "small" in graph_id or "medium" in graph_id:
            size_cat = "small"
        elif "2k" in graph_id:
            size_cat = "medium"
        elif "5k" in graph_id or "10k" in graph_id:
            size_cat = "large"
        elif "20k" in graph_id:
            size_cat = "very_large"
        else:
            size_cat = "unknown"
        
        # Process each method result
        for key, result in record.items():
            if key in ["task", "graph_id", "source", "target", "prompt"]:
                continue
            
            if not isinstance(result, dict):
                continue
            
            # Parse method type and model
            if key.startswith("tool_"):
                method_type = "tool_based"
                model_key = key[5:]  # Remove "tool_" prefix
            elif key.startswith("raw_"):
                method_type = "raw_llm"
                model_key = key[4:]  # Remove "raw_" prefix
            else:
                continue
            
            model_name = result.get("model_name", model_key)
            
            # Initialize model tracking
            if model_key not in analysis["models"]:
                analysis["models"][model_key] = {
                    "name": model_name,
                    "tool_based": {"total": 0, "correct": 0, "errors": 0},
                    "raw_llm": {"total": 0, "correct": 0, "errors": 0},
                }
            
            # Extract metrics
            metrics = result.get("metrics", {})
            is_correct = metrics.get("exact_match", False) if isinstance(metrics, dict) else False
            has_error = "error" in result or (isinstance(metrics, dict) and "error" in metrics)
            
            # Update counters
            model_data = analysis["models"][model_key][method_type]
            model_data["total"] += 1
            if is_correct:
                model_data["correct"] += 1
            if has_error:
                model_data["errors"] += 1
            
            # By task
            task_data = analysis["by_task"][task][f"{method_type}_{model_key}"]
            task_data["total"] += 1
            if is_correct:
                task_data["correct"] += 1
            if has_error:
                task_data["errors"] += 1
            
            # By graph size
            size_data = analysis["by_graph_size"][size_cat][f"{method_type}_{model_key}"]
            size_data["total"] += 1
            if is_correct:
                size_data["correct"] += 1
            if has_error:
                size_data["errors"] += 1
    
    # Calculate accuracies
    for model_key in analysis["models"]:
        for method_type in ["tool_based", "raw_llm"]:
            data = analysis["models"][model_key][method_type]
            data["accuracy"] = data["correct"] / data["total"] if data["total"] > 0 else 0
    
    # Calculate task and size accuracies
    for task_data in analysis["by_task"].values():
        for method_key, data in task_data.items():
            data["accuracy"] = data["correct"] / data["total"] if data["total"] > 0 else 0
    
    for size_data in analysis["by_graph_size"].values():
        for method_key, data in size_data.items():
            data["accuracy"] = data["correct"] / data["total"] if data["total"] > 0 else 0
    
    return analysis


def print_comparison_report(analysis: Dict):
    """Print comprehensive comparison report."""
    print("=" * 100)
    print("MULTI-MODEL EVALUATION COMPARISON REPORT")
    print("=" * 100)
    print()
    
    print(f"Total samples: {analysis['total_samples']}")
    print(f"Models evaluated: {len(analysis['models'])}")
    print()
    
    # Overall accuracy by model
    print("OVERALL ACCURACY BY MODEL:")
    print("-" * 100)
    print(f"{'Model':<30} {'Tool-Based':<20} {'Raw LLM':<20} {'Improvement':<20}")
    print("-" * 100)
    
    for model_key, model_data in sorted(analysis["models"].items()):
        name = model_data["name"]
        tool_data = model_data["tool_based"]
        raw_data = model_data["raw_llm"]
        
        tool_acc = tool_data["accuracy"] if tool_data["total"] > 0 else 0
        raw_acc = raw_data["accuracy"] if raw_data["total"] > 0 else 0
        improvement = tool_acc - raw_acc
        
        tool_str = f"{tool_acc:.1%} ({tool_data['correct']}/{tool_data['total']})"
        raw_str = f"{raw_acc:.1%} ({raw_data['correct']}/{raw_data['total']})" if raw_data["total"] > 0 else "N/A"
        imp_str = f"{improvement:+.1%}" if raw_data["total"] > 0 else "N/A"
        
        print(f"{name:<30} {tool_str:<20} {raw_str:<20} {imp_str:<20}")
    
    print()
    
    # By task type
    print("ACCURACY BY TASK TYPE (Tool-Based):")
    print("-" * 100)
    tasks = sorted(analysis["by_task"].keys())
    
    # Get all models
    models = list(analysis["models"].keys())
    
    print(f"{'Task':<20}", end="")
    for model_key in models:
        name = analysis["models"][model_key]["name"]
        print(f"{name[:15]:<20}", end="")
    print()
    print("-" * 100)
    
    for task in tasks:
        print(f"{task:<20}", end="")
        for model_key in models:
            key = f"tool_based_{model_key}"
            if key in analysis["by_task"][task]:
                data = analysis["by_task"][task][key]
                acc = data["accuracy"]
                result_str = f"{acc:.1%} ({data['correct']}/{data['total']})"
                print(f"{result_str:<20}", end="")
            else:
                print(f"{'N/A':<20}", end="")
        print()
    
    print()
    
    # By graph size
    print("ACCURACY BY GRAPH SIZE (Tool-Based):")
    print("-" * 100)
    sizes = ["small", "medium", "large", "very_large"]
    
    print(f"{'Size':<20}", end="")
    for model_key in models:
        name = analysis["models"][model_key]["name"]
        print(f"{name[:15]:<20}", end="")
    print()
    print("-" * 100)
    
    for size in sizes:
        if size not in analysis["by_graph_size"]:
            continue
        print(f"{size:<20}", end="")
        for model_key in models:
            key = f"tool_based_{model_key}"
            if key in analysis["by_graph_size"][size]:
                data = analysis["by_graph_size"][size][key]
                acc = data["accuracy"]
                result_str = f"{acc:.1%} ({data['correct']}/{data['total']})"
                print(f"{result_str:<20}", end="")
            else:
                print(f"{'N/A':<20}", end="")
        print()
    
    print()
    print("=" * 100)


def cli():
    parser = argparse.ArgumentParser(
        description="Analyze multi-model evaluation results."
    )
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Path to results JSON from eval_multi_model",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional: Save analysis to JSON",
    )
    args = parser.parse_args()
    
    # Load results
    with open(args.results) as f:
        results = json.load(f)
    
    # Analyze
    analysis = analyze_multi_model_results(results)
    
    # Print report
    print_comparison_report(analysis)
    
    # Save if requested
    if args.out:
        args.out.write_text(json.dumps(analysis, indent=2))
        print(f"\nAnalysis saved to {args.out}")


if __name__ == "__main__":
    cli()
