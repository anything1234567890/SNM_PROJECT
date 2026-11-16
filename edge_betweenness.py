"""
edge_betweenness.py
===================
Calculates Edge Betweenness Centrality on the actual road network,
ranks road segments according to their network importance,
displays the top 10 most critical roads, and exports the complete
ranking to a CSV file in the outputs/ directory.

STRICT FILE SAFETY:
- Standalone file: does not modify or overwrite any pre-existing project file.
- Preserves the real directed MultiDiGraph representation.
- Calculates weighted edge betweenness using verified physical road lengths.
- Preserves individual parallel edge keys and road identifiers.
"""

import ast
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import osmnx as ox
from osmnx.graph import _create_graph


# ============================================================================
# CONFIGURATION
# ============================================================================
DEFAULT_CACHE_JSON = Path("cache/2169d543b1cd5211fe3cb41b324894a706ad1961.json")
OUTPUT_DIR = Path("outputs")
DEFAULT_OUTPUT_CSV = OUTPUT_DIR / "road_edge_betweenness_ranking.csv"


# ============================================================================
# UTILITIES & DATA CLEANING
# ============================================================================
def clean_street_name(name: Any) -> str:
    """
    Cleans and standardizes OpenStreetMap road name tags.
    Handles nested lists, stringified Python lists, or missing tags.
    """
    if isinstance(name, list):
        cleaned_parts = [str(x).strip() for x in name if x]
        return " / ".join(cleaned_parts) if cleaned_parts else "Unnamed Road"

    if isinstance(name, str):
        stripped = name.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, list):
                    cleaned_parts = [str(x).strip() for x in parsed if x]
                    return " / ".join(cleaned_parts) if cleaned_parts else "Unnamed Road"
            except Exception:
                pass
        return stripped if stripped else "Unnamed Road"

    return "Unnamed Road"


def format_road_id(osmid: Any, u: Any, v: Any, key: Any) -> str:
    """
    Generates a clean, unique Road ID string preserving the OSM Way ID
    along with edge endpoints and multigraph key.
    """
    if isinstance(osmid, list):
        osm_str = "+".join(str(x) for x in osmid)
    elif osmid is not None:
        osm_str = str(osmid)
    else:
        osm_str = f"EDGE_{u}_{v}_{key}"
    return osm_str


# ============================================================================
# GRAPH LOADING (VERIFIED ACTUAL ROAD NETWORK)
# ============================================================================
def load_road_network(
    graphml_path: Optional[Path] = None,
    cache_json_path: Optional[Path] = None
) -> nx.MultiDiGraph:
    """
    Loads the actual road network as a directed MultiDiGraph.
    
    Priority:
    1. Specified GraphML file if exists.
    2. Verified cached Kapashera, Delhi road network JSON from cache/.
    """
    # 1. Try GraphML if specified and exists
    if graphml_path and Path(graphml_path).exists():
        print(f"Loading road network from GraphML: {graphml_path}")
        G = ox.load_graphml(graphml_path)
        if not isinstance(G, nx.MultiDiGraph):
            G = nx.MultiDiGraph(G)
        return G

    # 2. Load from the verified project cache
    json_path = cache_json_path or DEFAULT_CACHE_JSON
    if not json_path.exists():
        raise FileNotFoundError(
            f"Road network cache file not found at: {json_path.resolve()}"
        )

    print(f"Loading verified road network from cache: {json_path}")
    with open(json_path, "r", encoding="utf-8") as fp:
        raw_osm_data = json.load(fp)

    # Build directed MultiDiGraph from raw Overpass elements
    G_raw = _create_graph([raw_osm_data], bidirectional=False)
    print(f"Raw road network: {G_raw.number_of_nodes():,} nodes | {G_raw.number_of_edges():,} edges")

    # Simplify topology while preserving true geometric junctions
    print("Simplifying road network topology...")
    G_simplified = ox.simplify_graph(G_raw)

    # Isolate main connected component (same as graph_builder.py logic)
    print("Extracting primary connected road component...")
    undirected = G_simplified.to_undirected()
    main_component_nodes = max(nx.connected_components(undirected), key=len)
    G = G_simplified.subgraph(main_component_nodes).copy()

    print(f"Active road network: {G.number_of_nodes():,} nodes | {G.number_of_edges():,} edges")
    print(f"Graph representation: {type(G).__name__} (Directed Multigraph)")
    return G


# ============================================================================
# PHYSICAL ROAD LENGTH VALIDATION
# ============================================================================
def validate_edge_lengths(G: nx.MultiDiGraph) -> int:
    """
    Verifies that every edge has a valid, strictly positive physical length.
    If missing, calculates true geodesic great-circle distance between node
    coordinates rather than using arbitrary dummy weights.
    """
    fixed_count = 0
    for u, v, k, data in G.edges(keys=True, data=True):
        length = data.get("length")
        is_valid = False

        if length is not None:
            try:
                flen = float(length)
                if flen > 0:
                    data["length"] = flen
                    is_valid = True
            except (ValueError, TypeError):
                is_valid = False

        if not is_valid:
            # Physically calculate length using geodesic great-circle distance
            u_lat, u_lon = G.nodes[u].get("y"), G.nodes[u].get("x")
            v_lat, v_lon = G.nodes[v].get("y"), G.nodes[v].get("x")

            if None not in (u_lat, u_lon, v_lat, v_lon):
                calculated_len = ox.distance.great_circle(
                    float(u_lat), float(u_lon), float(v_lat), float(v_lon)
                )
                data["length"] = max(float(calculated_len), 0.1)
            else:
                geom = data.get("geometry")
                if geom and hasattr(geom, "length") and geom.length > 0:
                    data["length"] = float(geom.length)
                else:
                    raise ValueError(
                        f"Edge ({u}, {v}, {k}) lacks physical length and valid node coordinates."
                    )
            fixed_count += 1

    return fixed_count


# ============================================================================
# EDGE BETWEENNESS CENTRALITY COMPUTATION
# ============================================================================
def compute_edge_betweenness(
    G: nx.MultiDiGraph,
    weight: str = "length"
) -> Dict[Tuple[Any, Any, Any], float]:
    """
    Calculates Edge Betweenness Centrality on the directed MultiDiGraph
    weighted by physical road length.
    
    Formula:
        c_B(e) = sum_{s, t in V} (sigma(s, t | e) / sigma(s, t))
    """
    print(f"\nComputing Edge Betweenness Centrality (weighted by '{weight}')...")
    print("Evaluating shortest paths across all node pairs in the road network...")
    start_time = time.time()

    # NetworkX natively supports MultiDiGraphs, returning (u, v, key) -> score
    centrality_scores = nx.edge_betweenness_centrality(
        G,
        weight=weight,
        normalized=True
    )

    elapsed = time.time() - start_time
    print(f"Edge Betweenness Centrality calculation finished in {elapsed:.2f} seconds.")
    return centrality_scores


# ============================================================================
# ROAD RANKING & ENRICHMENT
# ============================================================================
def rank_roads(
    G: nx.MultiDiGraph,
    centrality_scores: Dict[Tuple[Any, Any, Any], float]
) -> List[Dict[str, Any]]:
    """
    Ranks all road edges in descending order of centrality score,
    preserving parallel edge keys and road metadata.
    """
    ranked_records = []

    # Sort edges strictly by centrality score in descending order
    sorted_edges = sorted(
        centrality_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    for rank, (edge_key_tuple, score) in enumerate(sorted_edges, start=1):
        u, v, key = edge_key_tuple
        edge_data = G.get_edge_data(u, v, key) or {}

        osmid = edge_data.get("osmid")
        road_id = format_road_id(osmid, u, v, key)
        raw_name = edge_data.get("name")
        clean_name = clean_street_name(raw_name)
        highway_type = edge_data.get("highway", "road")
        if isinstance(highway_type, list):
            highway_type = "/".join(str(x) for x in highway_type)
        length_m = float(edge_data.get("length", 0.0))
        oneway = bool(edge_data.get("oneway", False))

        ranked_records.append({
            "Rank": rank,
            "Road_ID": road_id,
            "Road_Name": clean_name,
            "Highway_Type": str(highway_type),
            "Source_Node": u,
            "Target_Node": v,
            "Edge_Key": key,
            "Centrality_Score": float(score),
            "Length_Meters": round(length_m, 2),
            "OneWay": oneway
        })

    return ranked_records


# ============================================================================
# CONSOLE DISPLAY
# ============================================================================
def display_top_roads(ranked_records: List[Dict[str, Any]], top_n: int = 10) -> None:
    """
    Renders an ASCII table displaying the top N most important roads.
    """
    top_records = ranked_records[:top_n]
    print("\n" + "=" * 115)
    print(f"               TOP {top_n} MOST IMPORTANT ROADS (EDGE BETWEENNESS CENTRALITY RANKING)")
    print("=" * 115)
    header = (
        f"{'Rank':<5} | {'Road Name':<30} | {'Highway':<14} | "
        f"{'Source -> Target (Key)':<30} | {'Score':<10} | {'Length (m)':<10}"
    )
    print(header)
    print("-" * 115)

    for r in top_records:
        endpoints_key = f"{r['Source_Node']}->{r['Target_Node']} (k={r['Edge_Key']})"
        road_name = (r['Road_Name'][:28] + "..") if len(r['Road_Name']) > 30 else r['Road_Name']
        highway = (r['Highway_Type'][:12] + "..") if len(r['Highway_Type']) > 14 else r['Highway_Type']
        endpoints_str = (endpoints_key[:28] + "..") if len(endpoints_key) > 30 else endpoints_key

        row = (
            f"{r['Rank']:<5} | {road_name:<30} | {highway:<14} | "
            f"{endpoints_str:<30} | {r['Centrality_Score']:<10.6f} | {r['Length_Meters']:<10.1f}"
        )
        print(row)

    print("=" * 115 + "\n")


# ============================================================================
# CSV EXPORT
# ============================================================================
def export_ranking_csv(
    ranked_records: List[Dict[str, Any]],
    output_path: Path
) -> Path:
    """
    Exports the complete road ranking to a new CSV file.
    Ensures no pre-existing file is overwritten.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prevent accidental overwrite
    target_path = output_path
    if target_path.exists():
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        target_path = output_path.parent / f"{output_path.stem}_{timestamp}{output_path.suffix}"
        print(f"Note: Output file exists. Writing to safe destination: {target_path}")

    fieldnames = [
        "Rank",
        "Road_ID",
        "Road_Name",
        "Highway_Type",
        "Source_Node",
        "Target_Node",
        "Edge_Key",
        "Centrality_Score",
        "Length_Meters",
        "OneWay"
    ]

    with open(target_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ranked_records)

    print(f"Complete road ranking exported successfully: {target_path.resolve()}")
    print(f"Total road segments exported: {len(ranked_records):,}")
    return target_path


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
def main():
    print("=" * 70)
    print(" ROAD NETWORK EDGE BETWEENNESS CENTRALITY & IMPORTANCE RANKING ")
    print("=" * 70)

    # 1. Load the actual road network graph
    G = load_road_network()

    # 2. Validate physical road lengths
    fixed_lengths = validate_edge_lengths(G)
    if fixed_lengths > 0:
        print(f"Physically recomputed lengths for {fixed_lengths} edge(s) using node geodesics.")
    else:
        print("Verified: 100% of road edges have positive, metric-accurate physical lengths.")

    # 3. Calculate Edge Betweenness Centrality
    centrality_scores = compute_edge_betweenness(G, weight="length")

    # 4. Rank roads
    ranked_roads = rank_roads(G, centrality_scores)

    # 5. Display top 10 most critical roads
    display_top_roads(ranked_roads, top_n=10)

    # 6. Export complete ranking to CSV
    export_ranking_csv(ranked_roads, DEFAULT_OUTPUT_CSV)

    print("Execution complete.")


if __name__ == "__main__":
    main()
