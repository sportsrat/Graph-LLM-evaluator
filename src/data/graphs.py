from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Dict, Iterable, List

import networkx as nx


BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"


@dataclass
class GraphSpec:
    graph_id: str
    path: pathlib.Path
    description: str


GRAPH_SPECS: Dict[str, GraphSpec] = {
    "ego-facebook-small": GraphSpec(
        graph_id="ego-facebook-small",
        path=DATA_DIR / "ego_facebook_small.edgelist",
        description="Toy subset of ego-Facebook graph (undirected).",
    ),
    "ego-facebook-medium": GraphSpec(
        graph_id="ego-facebook-medium",
        path=DATA_DIR / "ego_facebook_medium.edgelist",
        description="Medium subset of ego-Facebook graph (undirected).",
    ),
    "ca-grqc-small": GraphSpec(
        graph_id="ca-grqc-small",
        path=DATA_DIR / "ca_grqc_small.edgelist",
        description="Toy subset of ca-GrQc collaboration graph (undirected).",
    ),
    "soc-advogato-small": GraphSpec(
        graph_id="soc-advogato-small",
        path=DATA_DIR / "soc_advogato_small.edgelist",
        description="Toy subset of soc-advogato trust network (directed-ish but loaded undirected).",
    ),
    "gnp-2k": GraphSpec(
        graph_id="gnp-2k",
        path=DATA_DIR / "gnp_2k.edgelist",
        description="Erdős–Rényi G(n,p) graph with ~2000 nodes (undirected).",
    ),
    "gnp-5k": GraphSpec(
        graph_id="gnp-5k",
        path=DATA_DIR / "gnp_5k.edgelist",
        description="Erdős–Rényi G(n,p) graph with ~5000 nodes (undirected).",
    ),
    "gnp-10k": GraphSpec(
        graph_id="gnp-10k",
        path=DATA_DIR / "gnp_10k.edgelist",
        description="Erdős–Rényi G(n,p) graph with ~10000 nodes (undirected).",
    ),
    "gnp-20k": GraphSpec(
        graph_id="gnp-20k",
        path=DATA_DIR / "gnp_20k.edgelist",
        description="Erdős–Rényi G(n,p) graph with ~20000 nodes (undirected).",
    ),
}


def list_available_graphs() -> List[GraphSpec]:
    return list(GRAPH_SPECS.values())


def load_graph(graph_id: str) -> nx.Graph:
    if graph_id not in GRAPH_SPECS:
        raise ValueError(f"Unknown graph_id '{graph_id}'. Available: {list(GRAPH_SPECS)}")
    spec = GRAPH_SPECS[graph_id]
    if not spec.path.exists():
        raise FileNotFoundError(f"Edge list not found at {spec.path}")
    g = nx.read_edgelist(spec.path, nodetype=int)
    g = g.to_undirected()
    return g


def load_custom_edgelist(path: pathlib.Path, directed: bool = False) -> nx.Graph:
    graph = nx.read_edgelist(path, nodetype=int, create_using=nx.DiGraph() if directed else nx.Graph())
    return graph


def add_edges_from_iterable(graph: nx.Graph, edges: Iterable[tuple[int, int]]) -> None:
    graph.add_edges_from(edges)


def generate_gnp_graph(n: int, p: float, seed: int = 42) -> nx.Graph:
    """Generate an Erdős–Rényi G(n, p) graph for synthetic scaling tests."""
    return nx.gnp_random_graph(n=n, p=p, seed=seed)

