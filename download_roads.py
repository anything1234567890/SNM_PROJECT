
import os
import osmnx as ox

# 1. Create outputs folder
os.makedirs("outputs", exist_ok=True)

# 2. Select study area
place = "Faridabad, Haryana, India"

print(f"Downloading road network for {place}...")

# 3. Configure OSMnx
ox.settings.use_cache = True
ox.settings.log_console = True
ox.settings.timeout = 180

# 4. Download driving road network
G = ox.graph_from_place(
    place,
    network_type="drive",
    simplify=True,
    retain_all=True
)

# 5. Save road network
graph_path = "outputs/road_network.graphml"

ox.save_graphml(
    G,
    filepath=graph_path
)

# 6. Display information
print("\nRoad network downloaded successfully!")
print("Total nodes:", G.number_of_nodes())
print("Total edges:", G.number_of_edges())
print("Saved at:", graph_path)
