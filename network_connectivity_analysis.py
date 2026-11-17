
import os
import csv
import networkx as nx

GRAPH_PATH = "output_graph.graphml"
RANKING_PATH = "outputs/critical_road_ranking.csv"
OUTPUT_PATH = "outputs/connectivity_impact_analysis.csv"

FAILURE_LEVELS = [1, 5, 10, 25, 50, 100]


def load_road_network():
    if not os.path.exists(GRAPH_PATH):
        raise FileNotFoundError(
            "output_graph.graphml not found."
        )

    original_graph = nx.read_graphml(GRAPH_PATH)

    print("Road network loaded successfully")
    print("Nodes:", original_graph.number_of_nodes())
    print("Edges:", original_graph.number_of_edges())

    # Convert to an undirected graph for connectivity analysis.
    G = nx.Graph()

    G.add_nodes_from(original_graph.nodes(data=True))

    for u, v, data in original_graph.edges(data=True):
        if u == v:
            continue

        length = float(data.get("length", 1.0))

        if G.has_edge(u, v):
            G[u][v]["length"] = min(
                G[u][v]["length"], length
            )
        else:
            G.add_edge(u, v, length=length)

    return G


def load_critical_roads():
    if not os.path.exists(RANKING_PATH):
        raise FileNotFoundError(
            "Critical road ranking CSV not found."
        )

    roads = []

    with open(
        RANKING_PATH,
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            roads.append(row)

    roads.sort(
        key=lambda row: int(row["rank"])
    )

    return roads


def get_unique_critical_edges(G, roads):
    selected_edges = []
    seen_edges = set()

    for road in roads:
        u = str(road["u"])
        v = str(road["v"])

        edge_id = tuple(sorted((u, v)))

        if edge_id in seen_edges:
            continue

        if G.has_edge(u, v):
            selected_edges.append((u, v))
            seen_edges.add(edge_id)

    return selected_edges


def calculate_connectivity(G):
    components = list(nx.connected_components(G))

    component_count = len(components)

    largest_component_size = (
        max(len(component) for component in components)
        if components else 0
    )

    total_nodes = G.number_of_nodes()

    largest_component_percentage = (
        largest_component_size / total_nodes * 100
        if total_nodes else 0
    )

    # Calculate the percentage of node pairs
    # that can still reach each other.

    total_pairs = total_nodes * (total_nodes - 1) / 2

    reachable_pairs = sum(
        len(component) * (len(component) - 1) / 2
        for component in components
    )

    reachable_percentage = (
        reachable_pairs / total_pairs * 100
        if total_pairs else 100
    )

    isolated_nodes = len(list(nx.isolates(G)))

    return {
        "connected_components": component_count,
        "largest_component_nodes": largest_component_size,
        "largest_component_percentage": round(
            largest_component_percentage, 2
        ),
        "reachable_node_pairs_percentage": round(
            reachable_percentage, 2
        ),
        "isolated_nodes": isolated_nodes
    }


def run_impact_analysis(G, critical_edges):
    results = []

    total_nodes = G.number_of_nodes()

    print("\nCalculating original network connectivity...")

    baseline = calculate_connectivity(G)

    results.append({
        "scenario": "Baseline",
        "failed_road_segments": 0,
        "total_nodes": total_nodes,
        **baseline
    })

    print("\nOriginal Network:")
    print(baseline)

    for failure_count in FAILURE_LEVELS:

        if not critical_edges:
            break

        actual_count = min(
            failure_count,
            len(critical_edges)
        )

        print(
            f"\nSimulating failure of top "
            f"{actual_count} critical road segments..."
        )

        simulated_graph = G.copy()

        failed_edges = critical_edges[:actual_count]

        for u, v in failed_edges:
            if simulated_graph.has_edge(u, v):
                simulated_graph.remove_edge(u, v)

        metrics = calculate_connectivity(
            simulated_graph
        )

        results.append({
            "scenario": f"Top {actual_count} critical roads failed",
            "failed_road_segments": actual_count,
            "total_nodes": total_nodes,
            **metrics
        })

        print("Connected components:",
              metrics["connected_components"])

        print("Largest component:",
              metrics["largest_component_percentage"], "%")

        print("Reachable node pairs:",
              metrics["reachable_node_pairs_percentage"], "%")

        print("Isolated nodes:",
              metrics["isolated_nodes"])

    return results


def save_results(results):
    os.makedirs("outputs", exist_ok=True)

    if not results:
        print("No analysis results available.")
        return

    with open(
        OUTPUT_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys()
        )

        writer.writeheader()
        writer.writerows(results)

    print("\nAnalysis saved successfully!")
    print("Output:", OUTPUT_PATH)


def main():
    print("\n==========================================")
    print("   NETWORK CONNECTIVITY IMPACT ANALYSIS")
    print("==========================================")

    G = load_road_network()

    roads = load_critical_roads()

    critical_edges = get_unique_critical_edges(
        G, roads
    )

    print("\nUnique critical road segments:",
          len(critical_edges))

    results = run_impact_analysis(
        G, critical_edges
    )

    save_results(results)

    print("\n==========================================")
    print("       ANALYSIS COMPLETED")
    print("==========================================")


if __name__ == "__main__":
    main()
