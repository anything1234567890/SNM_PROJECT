import json
import networkx as nx
import osmnx as ox
import requests
from shapely.geometry import shape, Polygon
from satellite_extractor import fetch_satellite_image, extract_road_skeleton, skeleton_to_graph
from metrics_and_viz import calculate_graph_metrics, export_graph_image

def search_india_places(query_text: str, limit: int = 5):
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": query_text,
        "countrycodes": "in",
        "format": "json",
        "polygon_geojson": 1,
        "addressdetails": 1,
        "limit": limit
    }
    headers = {"User-Agent": "FloodSimGraphBuilder/1.0"}
    response = requests.get(url, params=params, headers=headers)
    response.raise_for_status()
    return response.json()

def build_graph_for_place(place_data: dict, output_path: str = "output_graph.graphml", preview_img: str = "graph_preview.png"):
    display_name = place_data["display_name"]
    print(f"\nProcessing region: {display_name}")
    
    bbox_raw = place_data.get("boundingbox")
    south, north, west, east = [float(coord) for coord in bbox_raw]
    bbox = [south, north, west, east]
    
    geojson_data = place_data.get("geojson")
    
    if geojson_data and geojson_data.get("type") in ["Polygon", "MultiPolygon"]:
        official_poly = shape(geojson_data)
        print("Official administrative boundary polygon detected.")
    else:
        print("No official closed polygon found; using official bounding box polygon.")
        official_poly = Polygon([(west, south), (east, south), (east, north), (west, north)])
        
    # Buffer polygon slightly (~250-300m) to preserve external connecting loops
    # 0.0025 degrees is ~275 meters at Delhi latitude
    buffered_poly = official_poly.buffer(0.0025)

    G = None
    custom_filter = (
        '["highway"~"motorway|trunk|primary|secondary|tertiary|'
        'unclassified|residential|service|track|living_street|path"]'
    )

    try:
        print("Attempting buffered vector road extraction (prevents severed boundary loops)...")
        # Extract graph with the buffered area so loop roads aren't chopped
        G_raw = ox.graph_from_polygon(
            buffered_poly,
            custom_filter=custom_filter,
            simplify=True,
            retain_all=True
        )
        
        # Keep the main connected components that intersect the true official region
        # This preserves roads connected via outer loops while eliminating true floating fragments
        undirected_raw = G_raw.to_undirected()
        main_component_nodes = max(nx.connected_components(undirected_raw), key=len)
        G = G_raw.subgraph(main_component_nodes).copy()
        print(f"Topological buffer reconnection complete. Connected main graph preserved.")

    except Exception as e:
        print(f"OSM vector road extraction failed: {e}")
        G = None

    # Fallback to satellite if extraction produced nothing
    min_edge_threshold = 10
    if G is None or len(G.edges) < min_edge_threshold:
        print("\n[Autonomous Fallback] OSM data sparse or unavailable. Switching to satellite imagery...")
        sat_img = fetch_satellite_image(bbox)
        skeleton = extract_road_skeleton(sat_img)
        G = skeleton_to_graph(skeleton, bbox)

    # Clean attributes for GraphML compatibility
    for _, _, data in G.edges(data=True):
        if "geometry" in data:
            del data["geometry"]
        for key, val in list(data.items()):
            if isinstance(val, list):
                data[key] = str(val)

    nx.write_graphml(G, output_path)
    print(f"\n[1/3] Successfully saved GraphML to '{output_path}'")

    print("[2/3] Rendering visual graph image...")
    export_graph_image(G, output_img_path=preview_img)

    print("\n[3/3] Calculating mathematical graph metrics...")
    metrics = calculate_graph_metrics(G, bbox)
    
    print("\n" + "=" * 50)
    print("         MATHEMATICAL ACCURACY REPORT")
    print("=" * 50)
    print(f" Total Nodes                  : {metrics['total_nodes']}")
    print(f" Total Edges                  : {metrics['total_edges']}")
    print(f" Total Road Length            : {metrics['total_length_km']} km")
    print(f" Road Density                 : {metrics['road_density_km_per_sq_km']} km/km²")
    print(f" Meshedness (Cycle Structure) : {metrics['meshedness_coefficient']}")
    print(f" Reachability (Main Cluster)  : {metrics['reachability_score_percent']}%")
    print(f" Disconnected Components      : {metrics['connected_components_count']}")
    print(f" Average Valency (Degree)     : {metrics['average_node_degree']}")
    print("-" * 50)
    print(f" OVERALL TOPOLOGICAL ACCURACY : {metrics['topological_accuracy_score']}%")
    print("=" * 50 + "\n")

    return G, metrics

if __name__ == "__main__":
    user_query = input("Enter Indian place name (e.g. Kapashera, Delhi): ")
    results = search_india_places(user_query)
    
    if not results:
        print("No location found. Please verify spelling.")
    else:
        print("\nMatching locations:")
        for idx, item in enumerate(results):
            print(f"[{idx}] {item['display_name']}")
            
        choice = int(input("\nSelect index: "))
        build_graph_for_place(results[choice], output_path="output_graph.graphml", preview_img="graph_preview.png")