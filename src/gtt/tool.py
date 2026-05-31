from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx


class GraphTraversalTool:
    """Lightweight Graph Traversal Tool (GTT) over a NetworkX graph."""

    def __init__(self, graph: nx.Graph, graph_id: str):
        self.graph = graph
        self.graph_id = graph_id
        self.call_log: List[Dict[str, Any]] = []

    def reset_log(self) -> None:
        """Clear call log before a new episode."""
        self.call_log.clear()

    def _log(self, action: str, args: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        entry = {"action": action, "args": args, "result_summary": {k: v for k, v in result.items() if k != "components"}}
        self.call_log.append(entry)
        return result

    def get_neighbors(self, node: Union[int, str]) -> Dict[str, Union[List, str]]:
        if node not in self.graph:
            return self._log("get_neighbors", {"node": node}, {"error": f"node {node} not in graph"})
        neighbors = list(self.graph.neighbors(node))
        return self._log("get_neighbors", {"node": node}, {"graph": self.graph_id, "node": node, "neighbors": neighbors})

    def check_connectivity(self, node_a: Union[int, str], node_b: Union[int, str]) -> Dict[str, Union[bool, str, List]]:
        if node_a not in self.graph or node_b not in self.graph:
            return self._log(
                "check_connectivity",
                {"node_a": node_a, "node_b": node_b},
                {"error": f"missing node(s): {node_a if node_a not in self.graph else ''} {node_b if node_b not in self.graph else ''}".strip()},
            )
        reachable = nx.has_path(self.graph, node_a, node_b)
        return self._log("check_connectivity", {"node_a": node_a, "node_b": node_b}, {"graph": self.graph_id, "node_a": node_a, "node_b": node_b, "connected": reachable})

    def run_bfs(self, start_node: Union[int, str], depth: int = 2) -> Dict[str, Union[List, str, int]]:
        if start_node not in self.graph:
            return self._log("run_bfs", {"start_node": start_node, "depth": depth}, {"error": f"node {start_node} not in graph"})
        visited = []
        for current_depth, layer in enumerate(nx.bfs_layers(self.graph, start_node)):
            if current_depth > depth:
                break
            visited.extend(layer)
        return self._log("run_bfs", {"start_node": start_node, "depth": depth}, {"graph": self.graph_id, "start": start_node, "depth": depth, "visited": visited})

    def shortest_path(self, source: Union[int, str], target: Union[int, str]) -> Dict[str, Union[List, str, int]]:
        if source not in self.graph or target not in self.graph:
            return self._log(
                "shortest_path",
                {"source": source, "target": target},
                {"error": f"missing node(s): {source if source not in self.graph else ''} {target if target not in self.graph else ''}".strip()},
            )
        try:
            path = nx.shortest_path(self.graph, source, target)
            return self._log(
                "shortest_path",
                {"source": source, "target": target},
                {"graph": self.graph_id, "source": source, "target": target, "path": path, "length": len(path) - 1},
            )
        except nx.NetworkXNoPath:
            return self._log(
                "shortest_path",
                {"source": source, "target": target},
                {"graph": self.graph_id, "source": source, "target": target, "error": "no path"},
            )

    def graph_diameter(self) -> Dict[str, Union[int, str, Tuple[int, int]]]:
        if not nx.is_connected(self.graph):
            # compute diameter on largest connected component
            largest_cc = max(nx.connected_components(self.graph), key=len)
            subgraph = self.graph.subgraph(largest_cc)
            diameter = nx.diameter(subgraph)
            return self._log(
                "graph_diameter",
                {},
                {
                "graph": self.graph_id,
                "note": "graph disconnected; diameter on largest component",
                "diameter": diameter,
                },
            )
        diameter = nx.diameter(self.graph)
        return self._log("graph_diameter", {}, {"graph": self.graph_id, "diameter": diameter})

    def connected_components(self) -> Dict[str, Union[int, List[List], str]]:
        comps = [list(c) for c in nx.connected_components(self.graph)]
        return self._log("connected_components", {}, {"graph": self.graph_id, "component_count": len(comps), "components": comps})

    def get_property(self, property_name: str) -> Dict[str, Union[int, str, float]]:
        property_name = property_name.lower()
        if property_name == "num_nodes":
            return self._log("get_property", {"property": property_name}, {"graph": self.graph_id, "num_nodes": self.graph.number_of_nodes()})
        if property_name == "num_edges":
            return self._log("get_property", {"property": property_name}, {"graph": self.graph_id, "num_edges": self.graph.number_of_edges()})
        if property_name == "average_degree":
            degrees = [d for _, d in self.graph.degree()]
            avg = sum(degrees) / len(degrees) if degrees else 0.0
            return self._log("get_property", {"property": property_name}, {"graph": self.graph_id, "average_degree": avg})
        return self._log("get_property", {"property": property_name}, {"error": f"unsupported property '{property_name}'"})

    def stats(self) -> Dict[str, Any]:
        return {"total_calls": len(self.call_log), "calls": self.call_log}

