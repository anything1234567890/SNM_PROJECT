
import csv
import os
import networkx as nx
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


GRAPH_FILE = "static/output_graph.graphml"
CSV_FILE = "critical_roads.csv"
OUTPUT_FILE = "static/critical_roads_map.png"


def main():
    # Load the generated road network
    G = nx.read_graphml(GRAPH_FILE)

    # Read the top critical roads
    critical_edges = set()

    with open(CSV_FILE, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            if int(row["rank"]) <= 20:
                u = row["road_node_u"]
                v = row["road_node_v"]
                critical_edges.add(frozenset((u, v)))

    print("Critical edges loaded:", len(critical_edges))
 
    # Get geographic coordinates from graph nodes
    positions = {}

    for node, data in G.nodes(data=True):
        x = float(data["x"])
        y = float(data["y"])
        positions[node] = (x, y)

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_facecolor("#111111")
    fig.patch.set_facecolor("#111111")

    # Draw all normal road segments in blue
    for u, v in G.edges():
        if u in positions and v in positions:
            x_values = [positions[u][0], positions[v][0]]
            y_values = [positions[u][1], positions[v][1]]

            ax.plot(
                x_values, y_values,
                color="#3399FF",
                linewidth=0.6,
                alpha=0.55,
                zorder=1
            )

     # Highlight top 20 critical road segments in red
    highlighted = 0

    print("Critical edges loaded:", len(critical_edges))
    print("Graph edges:", G.number_of_edges())

    for u, v in G.edges():
        if frozenset((u, v)) in critical_edges:
            highlighted += 1

            x_values = [positions[u][0], positions[v][0]]
            y_values = [positions[u][1], positions[v][1]]

            ax.plot(
                x_values,
                y_values,
                color="#FF3333",
                linewidth=3.0,
                alpha=1.0,
                zorder=5
            )

    print("Final highlighted count:", highlighted)

    ax.set_title(
        "Delhi Road Network - Critical Roads",
        color="white",
        fontsize=16
    )
    ax.axis("off")
    ax.set_aspect("equal", adjustable="datalim")

    os.makedirs("static", exist_ok=True)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_FILE,
        dpi=250,
        facecolor=fig.get_facecolor(),
        bbox_inches="tight"
    )
    plt.close(fig)

    print("Critical road map generated!")
    print(f"Critical roads highlighted: {highlighted}")
    print(f"Output saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
