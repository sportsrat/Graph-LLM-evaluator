from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agent import ToolUseAgent
from .agent_llm import ReActToolAgent
from .data.graphs import GRAPH_SPECS, load_graph
from .eval import evaluate_components, evaluate_diameter, evaluate_shortest_path
from .gtt.tool import GraphTraversalTool
from .models.llm import build_tiny_llama


def run_rule_based(task: str, graph_id: str, src: int | None, dst: int | None) -> dict:
    graph = load_graph(graph_id)
    gtt = GraphTraversalTool(graph, graph_id=graph_id)
    agent = ToolUseAgent(gtt)
    if task == "shortest_path":
        res = agent.shortest_path(src, dst)
        metrics = evaluate_shortest_path(graph, src, dst, res.answer.get("path"))
    elif task == "diameter":
        res = agent.graph_diameter()
        metrics = evaluate_diameter(graph, res.answer.get("diameter", res.answer.get("diameter")))
    else:
        res = agent.connected_components()
        metrics = evaluate_components(graph, res.answer.get("component_count"))
    return {"mode": "rule-based", "answer": res.answer, "metrics": metrics, "call_stats": res.call_stats}


def run_llm(task: str, graph_id: str, src: int | None, dst: int | None, backend: str) -> dict:
    graph = load_graph(graph_id)
    gtt = GraphTraversalTool(graph, graph_id=graph_id)
    if backend == "stub":
        agent = ReActToolAgent(gtt, use_stub=True)
    else:
        agent = ReActToolAgent(gtt, llm=build_tiny_llama())
    task_text = format_task_text(task, src, dst)
    res = agent.run(task_text)
    return {"mode": f"llm-{backend}", "result": res, "call_stats": gtt.stats()}


def format_task_text(task: str, src: int | None, dst: int | None) -> str:
    if task == "shortest_path":
        return f"Find the shortest path between node {src} and node {dst}."
    if task == "diameter":
        return "Estimate the graph diameter."
    return "Report the connected components."


def cli() -> None:
    parser = argparse.ArgumentParser(description="Run experiment with rule-based and LLM agents.")
    parser.add_argument("task", choices=["shortest_path", "diameter", "components"])
    parser.add_argument("--graph", default="ego-facebook-medium", choices=list(GRAPH_SPECS.keys()))
    parser.add_argument("--src", type=int)
    parser.add_argument("--dst", type=int)
    parser.add_argument("--backend", choices=["stub", "tiny-llama"], default="stub")
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    if args.task == "shortest_path" and (args.src is None or args.dst is None):
        raise SystemExit("src and dst required for shortest_path")

    rule = run_rule_based(args.task, args.graph, args.src, args.dst)
    llm = run_llm(args.task, args.graph, args.src, args.dst, backend=args.backend)
    payload = {"rule_based": rule, "llm": llm}
    print(json.dumps(payload, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"Wrote results to {args.out}")


if __name__ == "__main__":
    cli()

