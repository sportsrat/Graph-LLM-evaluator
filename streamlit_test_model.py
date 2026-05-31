#!/usr/bin/env python3
"""
Streamlit app for interactive model testing on graph reasoning tasks (Windows & CUDA/CPU compatible).
Features a sleek Pink + Black dark theme with clean, human-readable labels.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st
import torch

# Make project root importable (so we can import from `src`)
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure HF cache uses workspace directory on Windows to prevent C: drive OOM
if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = str(PROJECT_ROOT / ".cache" / "huggingface")

from src.data.graphs import GRAPH_SPECS, load_graph  # type: ignore
from src.eval_multi_model import evaluate_model_with_tools, evaluate_model_raw  # type: ignore
from src.models.llm import get_default_device, get_default_dtype  # type: ignore
from src.models.model_registry import (  # type: ignore
    AVAILABLE_MODELS,
    build_model,
    clear_model_cache,
    list_available_models,
)


@st.cache_resource(show_spinner=False)
def get_cached_st_model(model_key: str):
    """Cache loaded models across Streamlit interactions to avoid reloading weights."""
    return build_model(model_key)


# ---- Display Formatting Maps (No Underscores) ----

MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "stub": "Stub LLM (Instant)",
    "tinyllama-1.1b": "TinyLlama 1.1B",
    "qwen-0.5b": "Qwen2 0.5B",
    "qwen-1.5b": "Qwen2 1.5B",
    "qwen-7b": "Qwen2 7B",
    "mistral-7b": "Mistral 7B",
    "mistral-8x7b": "Mixtral 8x7B",
    "phi-2": "Phi 2",
    "gemma-2b": "Gemma 2B",
}

TASK_DISPLAY_NAMES: Dict[str, str] = {
    "shortest_path": "Shortest Path",
    "diameter": "Graph Diameter",
    "components": "Connected Components",
}

GRAPH_DISPLAY_NAMES: Dict[str, str] = {
    "ego-facebook-small": "Facebook Network (Small)",
    "ego-facebook-medium": "Facebook Network (Medium)",
    "ca-grqc-small": "GrQc Collaboration (Small)",
    "soc-advogato-small": "Advogato Trust Network (Small)",
    "gnp-2k": "Random Graph (2,000 Nodes)",
    "gnp-5k": "Random Graph (5,000 Nodes)",
    "gnp-10k": "Random Graph (10,000 Nodes)",
    "gnp-20k": "Random Graph (20,000 Nodes)",
}


def clean_label(text: str) -> str:
    """Format string into clean title case without underscores."""
    if text in MODEL_DISPLAY_NAMES:
        return MODEL_DISPLAY_NAMES[text]
    if text in TASK_DISPLAY_NAMES:
        return TASK_DISPLAY_NAMES[text]
    if text in GRAPH_DISPLAY_NAMES:
        return GRAPH_DISPLAY_NAMES[text]
    return text.replace("_", " ").replace("-", " ").title()


# ---- Helper functions ----


def get_task_prompt(task_type: str, graph_id: str, source: Optional[int] = None, target: Optional[int] = None) -> str:
    """Generate task prompt."""
    graph_clean = clean_label(graph_id)
    if task_type == "shortest_path":
        return f"Find the shortest path between node {source} and node {target} in graph '{graph_clean}'."
    elif task_type == "diameter":
        return f"Estimate the diameter of graph '{graph_clean}'."
    elif task_type == "components":
        return f"Report the connected components of graph '{graph_clean}'."
    return ""


def create_sample(
    task_type: str,
    graph_id: str,
    source: Optional[int] = None,
    target: Optional[int] = None,
) -> Dict[str, Any]:
    """Create a sample dict for evaluation."""
    return {
        "task": task_type,
        "graph_id": graph_id,
        "source": source,
        "target": target,
        "prompt": get_task_prompt(task_type, graph_id, source, target),
    }


def format_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten common metric fields into a clean human-readable dict for display."""
    flat: Dict[str, Any] = {}

    if "exact_match" in metrics:
        flat["Exact Match"] = bool(metrics["exact_match"])

    if {"pred_length", "true_length"}.issubset(metrics.keys()):
        flat["Predicted Path Length"] = metrics["pred_length"]
        flat["Ground Truth Path Length"] = metrics["true_length"]
        if "abs_error" in metrics:
            flat["Absolute Error"] = metrics["abs_error"]
    if {"pred_diameter", "true_diameter"}.issubset(metrics.keys()):
        flat["Predicted Graph Diameter"] = metrics["pred_diameter"]
        flat["Ground Truth Graph Diameter"] = metrics["true_diameter"]
    if {"pred_count", "true_count"}.issubset(metrics.keys()):
        flat["Predicted Component Count"] = metrics["pred_count"]
        flat["Ground Truth Component Count"] = metrics["true_count"]

    return flat


def render_result(title: str, result: Dict[str, Any], show_trace: bool = True) -> None:
    """Render an evaluation result in Streamlit."""
    st.subheader(title)

    if "error" in result:
        st.error(f"Error: {result['error']}")
        return

    metrics = result.get("metrics", {}) or {}
    if metrics:
        st.markdown("**Performance Metrics**")
        flat = format_metrics(metrics)
        for k, v in flat.items():
            if k == "Exact Match":
                status = "✅ Exact Match" if v else "❌ Incorrect"
                st.write(f"- **{k}**: {status}")
            else:
                st.write(f"- **{k}**: `{v}`")

    # Tool call stats
    if "call_stats" in result:
        stats = result["call_stats"]
        with st.expander("🛠️ Tool Call Statistics", expanded=False):
            st.write(f"**Total Calls**: `{stats.get('total_calls', 0)}`")
            calls = stats.get("calls") or []
            if calls:
                for i, call in enumerate(calls, 1):
                    action = clean_label(call.get("action", "unknown"))
                    args = call.get("args", {})
                    st.write(f"{i}. **{action}**({json.dumps(args)})")

    # Completion (for raw LLM)
    if "completion" in result:
        completion: str = result["completion"]
        with st.expander("📝 Raw Model Completion", expanded=False):
            st.text_area("Output Text", value=completion, height=180)

    # Extracted answer
    if "answer_extracted" in result:
        answer = result["answer_extracted"]
        clean_answer = {clean_label(str(k)): v for k, v in answer.items()} if isinstance(answer, dict) else answer
        with st.expander("🎯 Extracted Answer", expanded=True):
            st.json(clean_answer)

    # Trace (for tool-based)
    if show_trace and "trace" in result:
        trace = result["trace"]
        messages = trace.get("messages", []) if isinstance(trace, dict) else []
        if messages:
            with st.expander("🔍 Conversation Trace", expanded=False):
                for i, msg in enumerate(messages, 1):
                    st.text(msg)
                    if i >= 10:
                        remaining = len(messages) - i
                        if remaining > 0:
                            st.write(f"... and {remaining} more step messages")
                        break


# ---- Streamlit Custom Pink + Black Styling ----


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        /* Main background & typography */
        .stApp {
            background-color: #0d0d12;
            color: #f0f0f5;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: #12121a !important;
            border-right: 1px solid #ff2a8533;
        }

        /* Headings with pink glow */
        h1, h2, h3 {
            color: #ffffff !important;
            font-weight: 700 !important;
        }
        h1 {
            color: #ff2a85 !important;
            text-shadow: 0 0 12px rgba(255, 42, 133, 0.4);
        }

        /* Cards & containers */
        div[data-testid="stMetricValue"] {
            color: #ff2a85 !important;
            font-weight: 700 !important;
        }
        div[data-testid="stMetric"] {
            background-color: #181824;
            border: 1px solid #ff2a8533;
            border-radius: 10px;
            padding: 12px 16px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
        }

        /* Pink buttons */
        div.stButton > button {
            background: linear-gradient(135deg, #ff2a85 0%, #d60062 100%) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            padding: 10px 24px !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 15px rgba(255, 42, 133, 0.4) !important;
        }
        div.stButton > button:hover {
            background: linear-gradient(135deg, #ff4095 0%, #ff2a85 100%) !important;
            box-shadow: 0 6px 20px rgba(255, 42, 133, 0.7) !important;
            transform: translateY(-1px);
        }

        /* Selectboxes & inputs */
        div[data-baseweb="select"] > div {
            background-color: #181824 !important;
            border-color: #ff2a8544 !important;
            color: #ffffff !important;
        }

        /* Expanders */
        .streamlit-expanderHeader {
            background-color: #181824 !important;
            color: #ff80bf !important;
            border-radius: 8px !important;
            border: 1px solid #ff2a8522 !important;
        }

        /* Code block styling */
        pre, code {
            background-color: #14141e !important;
            border: 1px solid #ff2a8533 !important;
            color: #ff99ca !important;
            border-radius: 6px !important;
        }

        /* Divider lines */
        hr {
            border-color: #ff2a8533 !important;
        }

        /* Badges / Info boxes */
        div.stAlert {
            background-color: #1c1424 !important;
            border: 1px solid #ff2a8555 !important;
            color: #ffb3d9 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---- Main App ----


def main() -> None:
    st.set_page_config(
        page_title="Graph Reasoning Model Tester",
        page_icon="🌸",
        layout="wide",
    )

    inject_custom_css()

    st.title("🌸 Graph Reasoning Model Tester")
    st.write(
        "Compare **Tool-Augmented Reasoning (ReAct + GTT)** vs **Direct Raw LLM Reasoning** on graph benchmark tasks."
    )

    # Sidebar: System info & configuration
    st.sidebar.header("⚙️ Configuration")

    # Hardware detection
    detected_device = get_default_device().upper()
    cuda_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    device_label = f"CUDA ({cuda_name})" if detected_device == "CUDA" else "CPU (Windows)"
    st.sidebar.info(f"**Hardware Device**: {device_label}")

    if st.sidebar.button("🧹 Clear Model Cache"):
        clear_model_cache()
        st.cache_resource.clear()
        st.sidebar.success("Model cache cleared!")

    st.sidebar.markdown("---")

    # Models
    model_keys = list_available_models()
    model_choice = st.sidebar.selectbox(
        "Select Model",
        options=list(range(len(model_keys))),
        format_func=lambda i: f"{clean_label(model_keys[i])} ({AVAILABLE_MODELS[model_keys[i]].size})",
    )
    model_key = model_keys[model_choice]
    model_spec = AVAILABLE_MODELS[model_key]

    # Tasks
    task_keys = ["shortest_path", "diameter", "components"]
    task_choice = st.sidebar.selectbox(
        "Task Type",
        options=list(range(len(task_keys))),
        format_func=lambda i: clean_label(task_keys[i]),
    )
    task_type = task_keys[task_choice]

    # Graphs
    graph_ids = list(GRAPH_SPECS.keys())
    graph_choice = st.sidebar.selectbox(
        "Graph Dataset",
        options=list(range(len(graph_ids))),
        format_func=lambda i: clean_label(graph_ids[i]),
    )
    graph_id = graph_ids[graph_choice]

    # Load graph info
    try:
        graph = load_graph(graph_id)
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
    except Exception as e:
        graph = None
        num_nodes = 0
        num_edges = 0
        st.sidebar.error(f"Failed to load graph: {e}")

    # For shortest path, ask for source/target nodes
    source: Optional[int] = None
    target: Optional[int] = None
    if task_type == "shortest_path" and graph is not None:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Shortest Path Nodes")

        nodes = sorted(graph.nodes())
        if not nodes:
            st.sidebar.error("Graph has no nodes!")
        else:
            min_node, max_node = nodes[0], nodes[-1]
            default_target = nodes[1] if len(nodes) > 1 else max_node
            source = int(
                st.sidebar.number_input(
                    "Source Node",
                    min_value=int(min_node),
                    max_value=int(max_node),
                    value=int(min_node),
                    step=1,
                )
            )
            target = int(
                st.sidebar.number_input(
                    "Target Node",
                    min_value=int(min_node),
                    max_value=int(max_node),
                    value=int(default_target),
                    step=1,
                )
            )
            if source == target:
                st.sidebar.warning("Source and target should be different nodes.")

    st.markdown("---")

    # Show current configuration cards with clean names
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Model", value=clean_label(model_key), help=model_spec.description)
    with col2:
        st.metric(label="Task", value=clean_label(task_type))
    with col3:
        st.metric(label="Graph", value=clean_label(graph_id))
    with col4:
        st.metric(label="Graph Size", value=f"{num_nodes:,} Nodes / {num_edges:,} Edges")

    # Build sample and show prompt
    if task_type == "shortest_path" and (source is None or target is None):
        st.warning("Please choose valid source and target nodes in the sidebar.")
        return

    sample = create_sample(task_type, graph_id, source, target)

    with st.expander("📋 Task Prompt", expanded=True):
        st.code(sample["prompt"], language="markdown")

    # Run button
    run_clicked = st.button("🚀 Run Evaluation", type="primary")

    if not run_clicked:
        return

    # Run evaluations
    tool_result: Dict[str, Any] = {}
    raw_result: Dict[str, Any] = {}

    with st.spinner(f"Running evaluations on {clean_label(model_key)}..."):
        try:
            llm_instance = get_cached_st_model(model_key)
        except Exception as e:
            st.error(f"Failed to load model {clean_label(model_key)}: {e}")
            return

        try:
            tool_result = evaluate_model_with_tools(sample, model_key, max_steps=6, llm=llm_instance)
        except Exception as e:
            tool_result = {"error": f"Tool-based evaluation failed: {e}"}

        try:
            raw_result = evaluate_model_raw(sample, model_key, llm=llm_instance)
        except Exception as e:
            raw_result = {"error": f"Raw LLM evaluation failed: {e}"}

    # Show results side-by-side
    st.markdown("---")
    st.subheader("📊 Comparison Results")

    col_left, col_right = st.columns(2)
    with col_left:
        render_result("🛠️ Tool-Based Approach (GTT)", tool_result, show_trace=True)
    with col_right:
        render_result("🧠 Raw LLM Approach", raw_result, show_trace=False)

    # Summary
    tool_correct = bool(tool_result.get("metrics", {}).get("exact_match", False)) if "error" not in tool_result else False
    raw_correct = bool(raw_result.get("metrics", {}).get("exact_match", False)) if "error" not in raw_result else False

    st.markdown("---")
    st.subheader("📌 Summary")
    sum_col1, sum_col2 = st.columns(2)
    with sum_col1:
        st.write(f"**Model**: {clean_label(model_key)}")
        st.write(f"**Task**: {clean_label(task_type)}")
        st.write(f"**Graph**: {clean_label(graph_id)}")
    with sum_col2:
        st.write(f"**Tool-Based**: {'✅ Correct' if tool_correct else '❌ Incorrect or error'}")
        st.write(f"**Raw LLM**: {'✅ Correct' if raw_correct else '❌ Incorrect or error'}")


if __name__ == "__main__":
    main()

