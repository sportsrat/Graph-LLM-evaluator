#!/usr/bin/env python3
"""
Interactive model testing tool.

Allows you to select a model and test it on different graph reasoning tasks
with both tool-based and raw LLM approaches.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.graphs import GRAPH_SPECS, load_graph
from src.eval_multi_model import evaluate_model_with_tools, evaluate_model_raw
from src.models.model_registry import AVAILABLE_MODELS, list_available_models


def print_separator(char="=", length=80):
    """Print a separator line."""
    print(char * length)


def select_model():
    """Let user select a model."""
    print_separator()
    print("Available Models:")
    print_separator("-")
    
    models = list_available_models()
    for i, model_key in enumerate(models, 1):
        spec = AVAILABLE_MODELS[model_key]
        print(f"{i}. {model_key:<20} - {spec.name:<25} ({spec.size})")
    
    print()
    while True:
        try:
            choice = input(f"Select model (1-{len(models)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                return models[idx]
            else:
                print(f"Please enter a number between 1 and {len(models)}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nCancelled.")
            sys.exit(0)


def select_task():
    """Let user select a task type."""
    print_separator()
    print("Task Types:")
    print_separator("-")
    print("1. shortest_path - Find shortest path between two nodes")
    print("2. diameter      - Calculate graph diameter")
    print("3. components    - Count connected components")
    print()
    
    tasks = {
        "1": ("shortest_path", "Find shortest path"),
        "2": ("diameter", "Calculate diameter"),
        "3": ("components", "Count components"),
    }
    
    while True:
        try:
            choice = input("Select task (1-3): ").strip()
            if choice in tasks:
                task_type, description = tasks[choice]
                return task_type, description
            else:
                print("Please enter 1, 2, or 3")
        except KeyboardInterrupt:
            print("\nCancelled.")
            sys.exit(0)


def select_graph():
    """Let user select a graph."""
    print_separator()
    print("Available Graphs:")
    print_separator("-")
    
    graphs = list(GRAPH_SPECS.keys())
    for i, graph_id in enumerate(graphs, 1):
        spec = GRAPH_SPECS[graph_id]
        graph = load_graph(graph_id)
        print(f"{i}. {graph_id:<25} - {spec.description}")
        print(f"   ({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges)")
    
    print()
    while True:
        try:
            choice = input(f"Select graph (1-{len(graphs)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(graphs):
                return graphs[idx]
            else:
                print(f"Please enter a number between 1 and {len(graphs)}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nCancelled.")
            sys.exit(0)


def get_task_prompt(task_type: str, graph_id: str, source=None, target=None):
    """Generate task prompt."""
    if task_type == "shortest_path":
        return f"Find the shortest path between node {source} and node {target} in graph '{graph_id}'."
    elif task_type == "diameter":
        return f"Estimate the diameter of graph '{graph_id}'."
    elif task_type == "components":
        return f"Report the connected components of graph '{graph_id}'."
    return ""


def create_sample(task_type: str, graph_id: str, source=None, target=None):
    """Create a sample dict for evaluation."""
    return {
        "task": task_type,
        "graph_id": graph_id,
        "source": source,
        "target": target,
        "prompt": get_task_prompt(task_type, graph_id, source, target),
    }


def print_result(title: str, result: dict, show_prompts: bool = True):
    """Print evaluation result in a readable format."""
    print_separator()
    print(f"{title}")
    print_separator("-")
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return
    
    # Metrics
    metrics = result.get("metrics", {})
    print("Metrics:")
    if "exact_match" in metrics:
        match = "✓" if metrics["exact_match"] else "✗"
        print(f"  {match} Exact Match: {metrics['exact_match']}")
    
    # Task-specific metrics
    if "pred_length" in metrics and "true_length" in metrics:
        print(f"  Predicted Length: {metrics['pred_length']}")
        print(f"  True Length: {metrics['true_length']}")
        print(f"  Error: {metrics.get('abs_error', 'N/A')}")
    elif "pred_diameter" in metrics and "true_diameter" in metrics:
        print(f"  Predicted Diameter: {metrics['pred_diameter']}")
        print(f"  True Diameter: {metrics['true_diameter']}")
    elif "pred_count" in metrics and "true_count" in metrics:
        print(f"  Predicted Count: {metrics['pred_count']}")
        print(f"  True Count: {metrics['true_count']}")
    
    # Tool call stats (for tool-based)
    if "call_stats" in result:
        stats = result["call_stats"]
        print(f"\nTool Call Statistics:")
        print(f"  Total Calls: {stats.get('total_calls', 0)}")
        if stats.get("calls"):
            print(f"  Calls Made:")
            for i, call in enumerate(stats["calls"], 1):
                action = call.get("action", "unknown")
                args = call.get("args", {})
                print(f"    {i}. {action}({args})")
    
    # Completion (for raw LLM)
    if "completion" in result:
        completion = result["completion"]
        print(f"\nLLM Completion ({len(completion)} chars):")
        print(f"  {completion}")
    
    # Answer extracted
    if "answer_extracted" in result:
        answer = result["answer_extracted"]
        print(f"\nExtracted Answer:")
        print(f"  {json.dumps(answer, indent=2)}")
    
    # Trace (for tool-based)
    if "trace" in result and show_prompts:
        trace = result["trace"]
        messages = trace.get("messages", [])
        if messages:
            print(f"\nConversation Trace ({len(messages)} messages):")
            print_separator("-")
            
            for i, msg in enumerate(messages, 1):
                if "PROMPT_STEP" in msg:
                    step_num = msg.split("PROMPT_STEP_")[1].split(":")[0] if "PROMPT_STEP_" in msg else "?"
                    print(f"\n[{i}] PROMPT STEP {step_num}:")
                    # Try to extract and format messages
                    try:
                        # Find the messages list
                        if "'role': 'system'" in msg:
                            sys_start = msg.find("'system'")
                            sys_end = msg.find("}", sys_start)
                            if sys_end != -1:
                                sys_section = msg[sys_start:sys_end+50]
                                content_start = sys_section.find("'content':")
                                if content_start != -1:
                                    content = sys_section[content_start+11:].split("'")[0]
                                    print(f"  System: {content[:200]}{'...' if len(content) > 200 else ''}")
                        
                        if "'role': 'user'" in msg:
                            user_start = msg.find("'user'")
                            user_end = msg.find("}", user_start)
                            if user_end != -1:
                                user_section = msg[user_start:user_end+50]
                                content_start = user_section.find("'content':")
                                if content_start != -1:
                                    content = user_section[content_start+11:].split("'")[0]
                                    print(f"  User: {content[:200]}{'...' if len(content) > 200 else ''}")
                    except:
                        # Fallback: show raw message (truncated)
                        print(f"  {msg[:300]}...")
                        
                elif "LLM_STEP" in msg:
                    step_num = msg.split("LLM_STEP_")[1].split(":")[0] if "LLM_STEP_" in msg else "?"
                    print(f"\n[{i}] LLM RESPONSE STEP {step_num}:")
                    llm_part = msg.split("LLM_STEP_")[1].split("\n", 1)[1] if "\n" in msg else msg
                    print(f"  {llm_part[:400]}{'...' if len(llm_part) > 400 else ''}")
                
                # Limit output to first 6 messages to avoid overwhelming
                if i >= 6:
                    remaining = len(messages) - i
                    if remaining > 0:
                        print(f"\n  ... and {remaining} more messages")
                    break


def main():
    """Main interactive loop."""
    print_separator()
    print("Interactive Model Testing Tool")
    print("Test models on graph reasoning tasks")
    print_separator()
    
    # Select model
    model_key = select_model()
    model_name = AVAILABLE_MODELS[model_key].name
    print(f"\n✓ Selected: {model_name} ({model_key})")
    
    # Select task
    task_type, task_description = select_task()
    print(f"\n✓ Selected: {task_description}")
    
    # Select graph
    graph_id = select_graph()
    print(f"\n✓ Selected: {graph_id}")
    
    # Get source/target for shortest_path
    source, target = None, None
    if task_type == "shortest_path":
        graph = load_graph(graph_id)
        nodes = list(graph.nodes())
        print_separator()
        print(f"Available nodes: {nodes[:20]}{'...' if len(nodes) > 20 else ''}")
        while True:
            try:
                source = int(input(f"Enter source node: ").strip())
                if source in nodes:
                    break
                else:
                    print(f"Node {source} not in graph. Available: {nodes[:20]}")
            except ValueError:
                print("Please enter a valid number")
        while True:
            try:
                target = int(input(f"Enter target node: ").strip())
                if target in nodes and target != source:
                    break
                elif target == source:
                    print("Target must be different from source")
                else:
                    print(f"Node {target} not in graph. Available: {nodes[:20]}")
            except ValueError:
                print("Please enter a valid number")
    
    # Create sample
    sample = create_sample(task_type, graph_id, source, target)
    
    print_separator()
    print("Running Evaluations...")
    print_separator()
    
    # Show system prompt for tool-based
    from src.agent_llm import ReActToolAgent
    from src.gtt.tool import GraphTraversalTool
    graph = load_graph(graph_id)
    gtt = GraphTraversalTool(graph, graph_id=graph_id)
    temp_agent = ReActToolAgent(gtt, use_stub=True)
    system_prompt = temp_agent._build_system_prompt("")
    
    print("\n📋 System Prompt (Tool-Based):")
    print_separator("-")
    print(system_prompt)
    print()
    
    # Test tool-based approach
    print("\n🛠️  Testing Tool-Based Approach...")
    print(f"Task Prompt: {sample['prompt']}")
    print()
    try:
        tool_result = evaluate_model_with_tools(sample, model_key, max_steps=6)
        print_result("Tool-Based Results", tool_result, show_prompts=True)
    except Exception as e:
        print(f"❌ Error in tool-based evaluation: {e}")
        import traceback
        traceback.print_exc()
    
    # Show raw LLM prompt
    from src.eval_raw_llm import _build_raw_llm_prompt
    raw_prompt = _build_raw_llm_prompt(
        sample["prompt"],
        graph_id,
        graph,
        sample.get("source"),
        sample.get("target"),
    )
    
    print("\n\n🧠 Testing Raw LLM Approach...")
    print("Full Prompt (includes graph structure):")
    print_separator("-")
    print(raw_prompt[:800] + "..." if len(raw_prompt) > 800 else raw_prompt)
    print()
    try:
        raw_result = evaluate_model_raw(sample, model_key)
        print_result("Raw LLM Results", raw_result, show_prompts=False)
    except Exception as e:
        print(f"❌ Error in raw LLM evaluation: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    print_separator()
    print("Summary")
    print_separator("-")
    
    tool_correct = tool_result.get("metrics", {}).get("exact_match", False) if "error" not in tool_result else False
    raw_correct = raw_result.get("metrics", {}).get("exact_match", False) if "error" not in raw_result else False
    
    print(f"Model: {model_name}")
    print(f"Task: {task_description}")
    print(f"Graph: {graph_id}")
    print()
    print(f"Tool-Based: {'✓ Correct' if tool_correct else '✗ Incorrect'}")
    print(f"Raw LLM:    {'✓ Correct' if raw_correct else '✗ Incorrect'}")
    
    print_separator()
    print("Done!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCancelled by user.")
        sys.exit(0)
