#!/usr/bin/env python3
"""Summarize evaluation results comparing rule-based vs SFT."""
import json
from pathlib import Path
from collections import defaultdict

def summarize(results_path: Path):
    with open(results_path) as f:
        records = json.load(f)
    
    rule_stats = defaultdict(list)
    sft_stats = defaultdict(list)
    
    for rec in records:
        task = rec["task"]
        if "rule_based" in rec:
            rb = rec["rule_based"]
            rule_stats[task].append({
                "correct": rb["metrics"].get("exact_match", False),
                "tool_calls": rb["call_stats"].get("total_calls", 0),
            })
        if "sft" in rec:
            sft = rec["sft"]
            sft_stats[task].append({
                "correct": sft["metrics"].get("exact_match", False),
                "tool_calls": sft["call_stats"].get("total_calls", 0),
            })
    
    print("=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    
    for task in ["shortest_path", "diameter", "components"]:
        print(f"\n{task.upper()}:")
        if task in rule_stats:
            rb_samples = rule_stats[task]
            rb_acc = sum(s["correct"] for s in rb_samples) / len(rb_samples) * 100
            rb_avg_calls = sum(s["tool_calls"] for s in rb_samples) / len(rb_samples)
            print(f"  Rule-based: {rb_acc:.1f}% accuracy, {rb_avg_calls:.1f} avg tool calls ({len(rb_samples)} samples)")
        
        if task in sft_stats:
            sft_samples = sft_stats[task]
            sft_acc = sum(s["correct"] for s in sft_samples) / len(sft_samples) * 100
            sft_avg_calls = sum(s["tool_calls"] for s in sft_samples) / len(sft_samples)
            print(f"  SFT:        {sft_acc:.1f}% accuracy, {sft_avg_calls:.1f} avg tool calls ({len(sft_samples)} samples)")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results_sft.json"))
    args = parser.parse_args()
    summarize(args.results)
