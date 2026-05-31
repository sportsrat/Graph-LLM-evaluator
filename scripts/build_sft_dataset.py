from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.agent import ToolUseAgent  # noqa: E402
from src.data.graphs import load_graph  # noqa: E402
from src.data.tasks import TaskSample, load_tasks_jsonl  # noqa: E402
from src.gtt.tool import GraphTraversalTool  # noqa: E402


SYSTEM_PROMPT = (
    "You are a graph reasoning assistant. Use the Graph Traversal Tool (GTT) by emitting TOOL_CALL JSON, "
    "observe results, and finish with FINAL_ANSWER JSON. Keep thoughts concise."
)


def build_demo(sample: TaskSample) -> Dict[str, object]:
    graph = load_graph(sample.graph_id)
    gtt = GraphTraversalTool(graph, graph_id=sample.graph_id)
    agent = ToolUseAgent(gtt)

    if sample.task == "shortest_path":
        res = agent.shortest_path(sample.source, sample.target)
    elif sample.task == "diameter":
        res = agent.graph_diameter()
    else:
        res = agent.connected_components()

    thought_blocks: List[str] = []
    for step in res.steps:
        tool_json = json.dumps({"action": step.tool_call["action"], "args": step.tool_call["args"]})
        obs_json = json.dumps(step.observation)
        thought_blocks.append(f"THINK {step.thought}\nTOOL_CALL {tool_json}\nObservation: {obs_json}")
    final_json = json.dumps(res.answer)
    assistant = "\n".join(thought_blocks + [f"FINAL_ANSWER {final_json}"])

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": sample.prompt},
        {"role": "assistant", "content": assistant},
    ]
    return {
        "graph_id": sample.graph_id,
        "task": sample.task,
        "prompt": sample.prompt,
        "ground_truth": sample.ground_truth,
        "messages": messages,
        "tool_call_count": res.call_stats["total_calls"] if res.call_stats else len(res.steps),
    }


def cli() -> None:
    parser = argparse.ArgumentParser(description="Build SFT dataset (JSONL) with tool-call traces.")
    parser.add_argument("--tasks", type=Path, default=Path("data/tasks.jsonl"), help="Input tasks JSONL.")
    parser.add_argument("--out", type=Path, default=Path("data/sft_traces.jsonl"), help="Output JSONL for SFT.")
    args = parser.parse_args()

    task_samples = load_tasks_jsonl(args.tasks)
    payloads = [build_demo(ts) for ts in task_samples]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in payloads:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(payloads)} SFT traces to {args.out}")


if __name__ == "__main__":
    cli()

