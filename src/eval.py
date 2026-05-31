from __future__ import annotations

from typing import Dict, Optional

import networkx as nx


def evaluate_shortest_path(graph: nx.Graph, source: int, target: int, predicted_path) -> Dict[str, Optional[float]]:
    """Evaluate shortest path prediction.

    In addition to exact match, this returns:
    - abs_error: |pred_length - true_length|
    - rel_error: abs_error / true_length  (if true_length > 0)
    - within_1: True if abs_error <= 1 (tolerant correctness)
    """
    try:
        true_length = nx.shortest_path_length(graph, source, target)
    except nx.NetworkXNoPath:
        return {"exact_match": False, "error": "no_path_ground_truth"}

    if not predicted_path:
        return {"exact_match": False, "error": "no_predicted_path"}

    pred_length = len(predicted_path) - 1
    abs_error = abs(pred_length - true_length)
    rel_error: Optional[float] = None
    if true_length:
        rel_error = abs_error / float(true_length)

    return {
        "exact_match": pred_length == true_length,
        "pred_length": pred_length,
        "true_length": true_length,
        "abs_error": abs_error,
        "rel_error": rel_error,
        "within_1": abs_error <= 1,
    }


def evaluate_components(graph: nx.Graph, predicted_count: int) -> Dict[str, Optional[int]]:
    true_count = nx.number_connected_components(graph)
    abs_error: Optional[int] = None
    rel_error: Optional[float] = None
    if predicted_count is not None:
        abs_error = abs(predicted_count - true_count)
        if true_count:
            rel_error = abs_error / float(true_count)

    return {
        "exact_match": predicted_count == true_count,
        "pred_count": predicted_count,
        "true_count": true_count,
        "abs_error": abs_error,
        "rel_error": rel_error,
    }


def evaluate_diameter(graph: nx.Graph, predicted: int) -> Dict[str, Optional[int]]:
    if nx.is_connected(graph):
        true_diameter = nx.diameter(graph)
    else:
        largest_cc = max(nx.connected_components(graph), key=len)
        true_diameter = nx.diameter(graph.subgraph(largest_cc))
    abs_error: Optional[int] = None
    rel_error: Optional[float] = None
    if predicted is not None:
        abs_error = abs(predicted - true_diameter)
        if true_diameter:
            rel_error = abs_error / float(true_diameter)

    return {
        "exact_match": predicted == true_diameter,
        "pred_diameter": predicted,
        "true_diameter": true_diameter,
        "abs_error": abs_error,
        "rel_error": rel_error,
    }

