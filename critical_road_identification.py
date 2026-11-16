"""Rank structurally critical road segments using edge betweenness centrality."""
import argparse, csv, math
from pathlib import Path
import networkx as nx

def _length(data):
    try:
        x = float(data.get("length", 1.0))
        return x if math.isfinite(x) and x >= 0 else 1.0
    except (TypeError, ValueError):
        return 1.0

def load_undirected_road_graph(graphml_path):
    path = Path(graphml_path)
    if not path.is_file():
        raise FileNotFoundError(f"GraphML file not found: {path}")
    source = nx.read_graphml(path)
    graph = nx.Graph()
    graph.add_nodes_from(source.nodes(data=True))
    if source.is_multigraph():
        edges = ((u, v, data) for u, v, _key, data in source.edges(keys=True, data=True))
    else:
        edges = source.edges(data=True)
    for u, v, data in edges:
        if u == v:
            continue
        length = _length(data)
        if graph.has_edge(u, v):
            graph[u][v]["length"] = min(graph[u][v]["length"], length)
        else:
            graph.add_edge(u, v, length=length)
    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        raise ValueError("GraphML contains no usable road network.")
    return graph

def rank_critical_roads(graph, sample_size=None, seed=42):
    scores = nx.edge_betweenness_centrality(
        graph, k=sample_size, normalized=True, weight="length", seed=seed
    )
    rows = []
    for (u, v), score in scores.items():
        rows.append({"road_node_u": str(u), "road_node_v": str(v),
                     "length_m": float(graph[u][v].get("length", 1.0)),
                     "edge_betweenness": float(score)})
    rows.sort(key=lambda row: row["edge_betweenness"], reverse=True)
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return rows

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", default="output_graph.graphml")
    parser.add_argument("--output", default="critical_roads.csv")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--sample-size", type=int, default=None)
    args = parser.parse_args()
    if args.top < 1 or (args.sample_size is not None and args.sample_size < 1):
        parser.error("--top and --sample-size must be positive")
    graph = load_undirected_road_graph(args.graph)
    rows = rank_critical_roads(graph, args.sample_size)
    fields = ["rank", "road_node_u", "road_node_v", "length_m", "edge_betweenness"]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    print(f"Network: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    print(f"Ranked roads saved to {args.output}")
    for row in rows[:args.top]:
        print(f"{row['rank']}. {row['road_node_u']} -- {row['road_node_v']} | "
              f"edge betweenness={row['edge_betweenness']:.6f} | length={row['length_m']:.1f} m")

if __name__ == "__main__":
    main()
