import argparse
import pathlib
import sys
from pathlib import Path

# Ensure local src package is importable when running as a script
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))

from data.graphs import generate_gnp_graph


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic G(n,p) graph and save as edgelist.")
    parser.add_argument("--n", type=int, default=1000, help="Number of nodes")
    parser.add_argument("--p", type=float, default=0.01, help="Edge probability")
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("data/gnp_generated.edgelist"), help="Output path")
    args = parser.parse_args()

    g = generate_gnp_graph(args.n, args.p)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for u, v in g.edges():
            f.write(f"{u} {v}\n")
    print(f"Wrote graph with {g.number_of_nodes()} nodes, {g.number_of_edges()} edges to {args.out}")


if __name__ == "__main__":
    main()

