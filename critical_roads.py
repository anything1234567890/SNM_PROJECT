
import os
import pandas as pd
import networkx as nx

# 1. Actual graph path from your existing graph_builder.py
GRAPH_PATH = "output_graph.graphml"

# 2. Output folder
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 3. Load graph using NetworkX
# This supports string-based node IDs.
print("Loading road network...")

G = nx.read_graphml(
    GRAPH_PATH,
    node_type=str,
    force_multigraph=True
)

print("Total nodes:", G.number_of_nodes())
print("Total edges:", G.number_of_edges())

# 4. Convert to a simple undirected graph
G_undirected = nx.Graph(G)

print("Undirected graph created.")

# 5. Calculate Edge Betweenness Centrality
print("Calculating edge centrality...")

centrality = nx.edge_betweenness_centrality(
    G_undirected,
    normalized=True,
    weight=None
)

# 6. Store road information
road_data = []

for (u, v), score in centrality.items():

    edge_data = G_undirected[u][v]

    road_data.append({
        "u": u,
        "v": v,
        "road_name": edge_data.get("name", "Unknown"),
        "road_type": edge_data.get("highway", "Unknown"),
        "centrality": score
    })

# 7. Create DataFrame
df = pd.DataFrame(road_data)

# 8. Sort roads by centrality
df = df.sort_values(
    by="centrality",
    ascending=False
)

# 9. Select top 10 roads
top_10 = df.head(10)

# 10. Save result
output_path = os.path.join(
    OUTPUT_DIR,
    "top_10_critical_roads.csv"
)

top_10.to_csv(output_path, index=False)

print("\nTop 10 critical roads:")
print(top_10)

print("\nResults saved at:", output_path)
