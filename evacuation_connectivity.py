
import os
import pandas as pd
import networkx as nx
import osmnx as ox

# ---------------------------------------
# 1. File paths
# ---------------------------------------

GRAPH_PATH = "outputs/road_network.graphml"
VULNERABLE_PATH = "outputs/vulnerable_roads.csv"
LOCATIONS_PATH = "data/evacuation_locations.csv"

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------
# 2. Load road network
# ---------------------------------------

print("Loading road network...")

G = ox.load_graphml(GRAPH_PATH)

print("Total nodes:", G.number_of_nodes())
print("Total edges:", G.number_of_edges())


# ---------------------------------------
# 3. Load flood-exposed roads
# ---------------------------------------

print("\nLoading vulnerable roads...")

vulnerable = pd.read_csv(VULNERABLE_PATH)

print("Vulnerable roads:", len(vulnerable))


# ---------------------------------------
# 4. Load evacuation locations
# ---------------------------------------

print("\nLoading evacuation locations...")

locations = pd.read_csv(LOCATIONS_PATH)

required_columns = {"name", "type", "latitude", "longitude"}

if not required_columns.issubset(locations.columns):
    raise ValueError(
        "evacuation_locations.csv must contain: "
        "name, type, latitude, longitude"
    )

origins = locations[
    locations["type"].str.lower() == "origin"
]

destinations = locations[
    locations["type"].str.lower() == "destination"
]

if origins.empty or destinations.empty:
    raise ValueError(
        "Add at least one origin and one destination "
        "to evacuation_locations.csv"
    )


# ---------------------------------------
# 5. Create normal and flood graphs
# ---------------------------------------

print("\nPreparing road networks...")

normal_graph = ox.convert.to_undirected(G)

flood_graph = normal_graph.copy()


# ---------------------------------------
# 6. Remove flood-exposed roads
# ---------------------------------------

print("Closing flood-exposed roads...")

# Convert CSV node IDs to match graph node IDs
sample_node = next(iter(flood_graph.nodes))

def convert_node_id(value):
    if isinstance(sample_node, int):
        return int(value)
    return str(value)


removed_edges = 0

for row in vulnerable.itertuples(index=False):

    u = convert_node_id(row.u)
    v = convert_node_id(row.v)

    if not flood_graph.has_edge(u, v):
        continue

    # If key exists in CSV, remove the matching edge.
    # Otherwise, remove all parallel edges between u and v.
    if "key" in vulnerable.columns and pd.notna(row.key):

        key = int(row.key)

        if flood_graph.has_edge(u, v, key):
            flood_graph.remove_edge(u, v, key)
            removed_edges += 1

    else:
        edge_keys = list(
            flood_graph[u][v].keys()
        )

        for key in edge_keys:
            flood_graph.remove_edge(u, v, key)
            removed_edges += 1


print("Flood-exposed edges removed:", removed_edges)


# ---------------------------------------
# 7. Find nearest road-network node
# ---------------------------------------

print("\nFinding nearest road nodes...")

def nearest_node(latitude, longitude):

    return ox.distance.nearest_nodes(
        G,
        X=float(longitude),
        Y=float(latitude)
    )


# ---------------------------------------
# 8. Calculate evacuation connectivity
# ---------------------------------------

print("\nStarting evacuation analysis...")

results = []

for origin in origins.itertuples(index=False):

    start = nearest_node(
        origin.latitude,
        origin.longitude
    )

    for destination in destinations.itertuples(index=False):

        end = nearest_node(
            destination.latitude,
            destination.longitude
        )

        # Normal road distance
        try:
            normal_distance = nx.shortest_path_length(
                normal_graph,
                source=start,
                target=end,
                weight="length"
            )

        except nx.NetworkXNoPath:
            normal_distance = None

        # Distance after flood-related road closures
        try:
            flood_distance = nx.shortest_path_length(
                flood_graph,
                source=start,
                target=end,
                weight="length"
            )

        except nx.NetworkXNoPath:
            flood_distance = None

        # Calculate additional distance
        if (
            normal_distance is not None
            and flood_distance is not None
        ):
            additional_distance = (
                flood_distance - normal_distance
            )
        else:
            additional_distance = None

        results.append({
            "origin": origin.name,
            "destination": destination.name,
            "normal_distance_m": normal_distance,
            "flood_distance_m": flood_distance,
            "additional_distance_m": additional_distance,
            "reachable_after_flood": (
                flood_distance is not None
            )
        })


# ---------------------------------------
# 9. Save results
# ---------------------------------------

results_df = pd.DataFrame(results)

output_path = os.path.join(
    OUTPUT_DIR,
    "connectivity_results.csv"
)

results_df.to_csv(output_path, index=False)


# ---------------------------------------
# 10. Display results
# ---------------------------------------

print("\nEvacuation connectivity results:")

print(results_df.to_string(index=False))

print("\nAnalysis completed successfully!")

print("Flood-exposed edges removed:", removed_edges)

print("Results saved at:", output_path)
