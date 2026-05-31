from __future__ import annotations

import json
import re
from typing import Dict, Optional

import networkx as nx

try:
    from .data.graphs import load_graph
    from .eval import evaluate_components, evaluate_diameter, evaluate_shortest_path
    from .models.llm import TransformersLLM, build_tiny_llama, build_from_checkpoint
except ImportError:
    # For direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from src.data.graphs import load_graph
    from src.eval import evaluate_components, evaluate_diameter, evaluate_shortest_path
    from src.models.llm import TransformersLLM, build_tiny_llama, build_from_checkpoint


def _format_graph_for_prompt(graph: nx.Graph, max_nodes: int = 50) -> str:
    """Format graph structure as text for LLM prompt. Only include if graph is small enough."""
    if graph.number_of_nodes() > max_nodes:
        return f"[Graph too large: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges. Cannot include full structure in context.]"
    
    # Format as edge list - include ALL edges (no truncation)
    edges = list(graph.edges())
    edge_list = "\n".join([f"  {u} -- {v}" for u, v in edges])
    
    # Include ALL nodes (no truncation)
    nodes = list(graph.nodes())
    
    return f"""Graph structure:
Nodes: {nodes} (total: {graph.number_of_nodes()} nodes)
Edges:
{edge_list}"""


def _build_raw_llm_prompt(task: str, graph_id: str, graph: nx.Graph, source: Optional[int] = None, target: Optional[int] = None) -> str:
    """Build a prompt that includes graph structure for raw LLM reasoning."""
    graph_info = _format_graph_for_prompt(graph)
    
    prompt = f"""You are a graph reasoning assistant. Given the graph structure below, answer the question.

{graph_info}

Task: {task}

Please reason step by step and provide your answer in JSON format:
- For shortest path: {{"path": [node1, node2, ...], "length": N}}
- For diameter: {{"diameter": N}}
- For connected components: {{"component_count": N}}

Answer:"""
    
    return prompt


def _extract_answer_from_raw_llm(completion: str, task: str) -> Optional[Dict]:
    """Extract structured answer from raw LLM completion."""
    # Try to find JSON in the completion
    json_patterns = [
        r'\{[^{}]*"path"[^{}]*\}',  # Path JSON
        r'\{[^{}]*"diameter"[^{}]*\}',  # Diameter JSON
        r'\{[^{}]*"component_count"[^{}]*\}',  # Components JSON
        r'\{.*?\}',  # Any JSON object
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, completion, re.DOTALL)
        for match in matches:
            try:
                answer = json.loads(match)
                # Validate it has the right structure
                if task == "shortest_path" and "path" in answer:
                    return answer
                elif task == "diameter" and ("diameter" in answer or "length" in answer):
                    return answer
                elif task == "components" and ("component_count" in answer or "count" in answer):
                    return answer
            except json.JSONDecodeError:
                continue
    
    # Fallback: try to extract numbers or paths from text
    if task == "shortest_path":
        # Look for path patterns like [0, 1, 2] or "0 -> 1 -> 2"
        path_patterns = [
            r'\[(\d+(?:\s*,\s*\d+)*)\]',  # [0, 1, 2]
            r'(\d+(?:\s*->\s*\d+)+)',  # 0 -> 1 -> 2
        ]
        for pattern in path_patterns:
            match = re.search(pattern, completion)
            if match:
                path_str = match.group(1)
                # Extract numbers
                nodes = [int(n.strip()) for n in re.findall(r'\d+', path_str)]
                if len(nodes) >= 2:
                    return {"path": nodes, "length": len(nodes) - 1}
    
    elif task == "diameter":
        # Look for diameter number
        diameter_match = re.search(r'diameter[:\s]+(\d+)', completion, re.IGNORECASE)
        if diameter_match:
            return {"diameter": int(diameter_match.group(1))}
    
    elif task == "components":
        # Look for component count
        count_match = re.search(r'(?:component|connected)[\s\w]*count[:\s]+(\d+)', completion, re.IGNORECASE)
        if count_match:
            return {"component_count": int(count_match.group(1))}
    
    return None


def evaluate_raw_llm(
    sample: Dict,
    checkpoint: Optional[str] = None,
    max_new_tokens: int = 512,
) -> Dict:
    """
    Evaluate raw LLM performance (without tools) by including graph structure in prompt.
    
    For small graphs: Includes full graph structure
    For large graphs: Includes summary (will likely fail or be inaccurate)
    """
    graph = load_graph(sample["graph_id"])
    
    # Build prompt with graph structure
    prompt = _build_raw_llm_prompt(
        sample["prompt"],
        sample["graph_id"],
        graph,
        sample.get("source"),
        sample.get("target"),
    )
    
    # Load LLM
    try:
        if checkpoint:
            llm = build_from_checkpoint(checkpoint)
        else:
            llm = build_tiny_llama()
        checkpoint_name = checkpoint or "base-model"
    except Exception as e:
        print(f"Warning: Failed to load LLM: {e}")
        llm = build_tiny_llama()
        checkpoint_name = "base-model"
    
    # Generate response
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
    except Exception as e:
        return {
            "mode": f"raw-llm-{checkpoint_name}",
            "error": str(e),
            "metrics": {"correct": False, "error": "generation_failed"},
        }
    
    # Extract answer
    answer = _extract_answer_from_raw_llm(completion, sample["task"])
    
    if answer is None:
        # Try to evaluate the raw completion as-is
        answer = {"raw_completion": completion}  # Store full completion
    
    # Evaluate
    metrics = _eval_sample(
        answer,
        graph,
        sample["task"],
        sample.get("source"),
        sample.get("target"),
    )
    
    return {
        "mode": f"raw-llm-{checkpoint_name}",
        "metrics": metrics,
        "completion": completion,  # Store full completion
        "answer_extracted": answer,
        "graph_size": {"nodes": graph.number_of_nodes(), "edges": graph.number_of_edges()},
    }


def _eval_sample(answer_raw: object, graph, task: str, src: Optional[int] = None, dst: Optional[int] = None) -> Dict:
    """Evaluate extracted answer against ground truth."""
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


if __name__ == "__main__":
    # Test with a sample
    import sys
    from pathlib import Path
    
    sys.path.append(str(Path(__file__).parent.parent))
    
    from datasets import load_dataset
    
    data = load_dataset("json", data_files="data/eval_tasks.jsonl")["train"]
    sample = data[0]
    
    print("Testing raw LLM evaluation...")
    result = evaluate_raw_llm(sample)
    print(json.dumps(result, indent=2))
