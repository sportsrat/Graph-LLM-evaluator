#!/usr/bin/env python3
"""
Compare raw LLM performance vs tool-based approach.

This script runs both evaluation methods and generates a comparison report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from datasets import load_dataset


def analyze_results(results: List[Dict]) -> Dict:
    """Analyze and compare results from different methods."""
    analysis = {
        "total_samples": len(results),
        "methods": {},
        "by_task": {},
        "by_graph_size": {},
    }
    
    # Initialize counters
    methods = set()
    tasks = set()
    graph_sizes = {}
    
    for record in results:
        graph_id = record.get("graph_id", "unknown")
        task = record.get("task", "unknown")
        tasks.add(task)
        
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
        
        graph_sizes[graph_id] = size_cat
        
        # Process each method
        for method_key in ["rule_based", "sft", "rlvr", "raw_llm"]:
            if method_key not in record:
                continue
            
            methods.add(method_key)
            method_data = record[method_key]
            
            if method_key not in analysis["methods"]:
                analysis["methods"][method_key] = {
                    "total": 0,
                    "correct": 0,
                    "errors": 0,
                    "by_task": {},
                    "by_graph_size": {},
                }
            
            # Extract metrics
            metrics = method_data.get("metrics", {})
            is_correct = metrics.get("exact_match", False) if isinstance(metrics, dict) else False
            
            analysis["methods"][method_key]["total"] += 1
            if is_correct:
                analysis["methods"][method_key]["correct"] += 1
            elif metrics.get("error"):
                analysis["methods"][method_key]["errors"] += 1
            
            # By task
            if task not in analysis["methods"][method_key]["by_task"]:
                analysis["methods"][method_key]["by_task"][task] = {"total": 0, "correct": 0}
            analysis["methods"][method_key]["by_task"][task]["total"] += 1
            if is_correct:
                analysis["methods"][method_key]["by_task"][task]["correct"] += 1
            
            # By graph size
            if size_cat not in analysis["methods"][method_key]["by_graph_size"]:
                analysis["methods"][method_key]["by_graph_size"][size_cat] = {"total": 0, "correct": 0}
            analysis["methods"][method_key]["by_graph_size"][size_cat]["total"] += 1
            if is_correct:
                analysis["methods"][method_key]["by_graph_size"][size_cat]["correct"] += 1
    
    # Calculate accuracies
    for method_key in analysis["methods"]:
        method = analysis["methods"][method_key]
        method["accuracy"] = method["correct"] / method["total"] if method["total"] > 0 else 0
        
        for task in method["by_task"]:
            task_data = method["by_task"][task]
            task_data["accuracy"] = task_data["correct"] / task_data["total"] if task_data["total"] > 0 else 0
        
        for size_cat in method["by_graph_size"]:
            size_data = method["by_graph_size"][size_cat]
            size_data["accuracy"] = size_data["correct"] / size_data["total"] if size_data["total"] > 0 else 0
    
    return analysis


def print_comparison_report(analysis: Dict):
    """Print a human-readable comparison report."""
    print("=" * 80)
    print("RAW LLM vs TOOL-BASED APPROACH COMPARISON")
    print("=" * 80)
    print()
    
    print(f"Total samples evaluated: {analysis['total_samples']}")
    print()
    
    # Overall accuracy comparison
    print("OVERALL ACCURACY:")
    print("-" * 80)
    if "rule_based" in analysis["methods"]:
        rb = analysis["methods"]["rule_based"]
        print(f"Rule-Based (baseline):     {rb['accuracy']:.1%} ({rb['correct']}/{rb['total']})")
    
    if "raw_llm" in analysis["methods"]:
        raw = analysis["methods"]["raw_llm"]
        print(f"Raw LLM (no tools):        {raw['accuracy']:.1%} ({raw['correct']}/{raw['total']})")
    
    if "sft" in analysis["methods"]:
        sft = analysis["methods"]["sft"]
        print(f"SFT + Tools:               {sft['accuracy']:.1%} ({sft['correct']}/{sft['total']})")
    
    if "rlvr" in analysis["methods"]:
        rlvr = analysis["methods"]["rlvr"]
        print(f"RLVR + Tools:              {rlvr['accuracy']:.1%} ({rlvr['correct']}/{rlvr['total']})")
    
    print()
    
    # By task type
    if "raw_llm" in analysis["methods"]:
        print("ACCURACY BY TASK TYPE (Raw LLM vs Tool-Based):")
        print("-" * 80)
        raw = analysis["methods"]["raw_llm"]
        
        for task in ["shortest_path", "diameter", "components"]:
            if task in raw["by_task"]:
                raw_task = raw["by_task"][task]
                print(f"\n{task.upper()}:")
                print(f"  Raw LLM:        {raw_task['accuracy']:.1%} ({raw_task['correct']}/{raw_task['total']})")
                
                # Compare with tool-based if available
                for method_key in ["sft", "rlvr"]:
                    if method_key in analysis["methods"] and task in analysis["methods"][method_key]["by_task"]:
                        tool_task = analysis["methods"][method_key]["by_task"][task]
                        improvement = tool_task["accuracy"] - raw_task["accuracy"]
                        print(f"  {method_key.upper()} + Tools: {tool_task['accuracy']:.1%} ({tool_task['correct']}/{tool_task['total']}) "
                              f"[{improvement:+.1%} improvement]")
    
    print()
    
    # By graph size
    if "raw_llm" in analysis["methods"]:
        print("ACCURACY BY GRAPH SIZE (Raw LLM vs Tool-Based):")
        print("-" * 80)
        raw = analysis["methods"]["raw_llm"]
        
        for size_cat in ["small", "medium", "large", "very_large"]:
            if size_cat in raw["by_graph_size"]:
                raw_size = raw["by_graph_size"][size_cat]
                print(f"\n{size_cat.upper()} graphs:")
                print(f"  Raw LLM:        {raw_size['accuracy']:.1%} ({raw_size['correct']}/{raw_size['total']})")
                
                # Compare with tool-based if available
                for method_key in ["sft", "rlvr"]:
                    if method_key in analysis["methods"] and size_cat in analysis["methods"][method_key]["by_graph_size"]:
                        tool_size = analysis["methods"][method_key]["by_graph_size"][size_cat]
                        improvement = tool_size["accuracy"] - raw_size["accuracy"]
                        print(f"  {method_key.upper()} + Tools: {tool_size['accuracy']:.1%} ({tool_size['correct']}/{tool_size['total']}) "
                              f"[{improvement:+.1%} improvement]")
    
    print()
    print("=" * 80)


def cli():
    parser = argparse.ArgumentParser(
        description="Compare raw LLM vs tool-based approach from evaluation results."
    )
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Path to results JSON file from eval_pipeline",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional: Save analysis to JSON file",
    )
    args = parser.parse_args()
    
    # Load results
    with open(args.results) as f:
        results = json.load(f)
    
    # Analyze
    analysis = analyze_results(results)
    
    # Print report
    print_comparison_report(analysis)
    
    # Save if requested
    if args.out:
        args.out.write_text(json.dumps(analysis, indent=2))
        print(f"\nAnalysis saved to {args.out}")


if __name__ == "__main__":
    cli()
