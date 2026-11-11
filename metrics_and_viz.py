import networkx as nx
import matplotlib
matplotlib.use("Agg")  # Thread-safe headless backend for FastAPI / Uvicorn
import matplotlib.pyplot as plt
import networkx as nx


def calculate_graph_metrics(G, bbox):
    """
    Computes rigorous topological accuracy, meshedness, and structural validity metrics.
    """
    south, north, west, east = bbox
    lat_dist = (north - south) * 111.0
    lon_dist = (east - west) * 111.0 * 0.88
    area_sq_km = max(lat_dist * lon_dist, 0.01)

    total_edges = len(G.edges)
    total_nodes = len(G.nodes)

    if total_nodes == 0:
        return {"error": "Empty graph"}

    total_length_m = sum(data.get("length", 0) for _, _, data in G.edges(data=True))
    total_length_km = total_length_m / 1000.0

    undirected_G = G.to_undirected()
    components = list(nx.connected_components(undirected_G))
    c = len(components)
    largest_comp_size = len(max(components, key=len)) if components else 0
    reachability_ratio = (largest_comp_size / total_nodes)

    # 1. Meshedness Coefficient (alpha) for planar road structures
    # Expected real road network range: 0.10 - 0.35
    v = total_nodes
    e = total_edges
    if v > 2:
        alpha = (e - v + c) / max((2 * v - 5), 1)
        alpha = max(0.0, min(alpha, 1.0))
    else:
        alpha = 0.0

    # 2. Realistic Degree Balance (Degree 1, 3, and 4 vs Degree 2 and >4)
    degrees = dict(undirected_G.degree())
    avg_degree = sum(degrees.values()) / v if v > 0 else 0
    
    # Penalize graphs that deviate heavily from typical road network meshedness (~0.20)
    # If alpha > 0.40, it's over-segmented (mesh/honeycomb noise)
   # Urban Indian localities frequently have dense grid/alley networks (alpha up to 0.70)
    if alpha <= 0.70:
        meshedness_quality = 1.0
    else:
        meshedness_quality = max(0.0, 1.0 - (alpha - 0.70) * 2.0)

    # 3. Composite Topological Accuracy Score (0 - 100%)
    # Blends component reachability, planar meshedness quality, and reasonable road density
    density = total_length_km / area_sq_km
    # Target density range for urban/semi-urban India: 8 - 25 km/km²
    if density <= 25.0:
        density_score = min(1.0, density / 12.0)
    else:
        density_score = max(0.0, 1.0 - (density - 25.0) / 35.0)

    accuracy_score = (
        0.40 * reachability_ratio +
        0.35 * meshedness_quality +
        0.25 * density_score
    ) * 100.0

    metrics = {
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "total_length_km": round(total_length_km, 2),
        "road_density_km_per_sq_km": round(density, 2),
        "meshedness_coefficient": round(alpha, 3),
        "connected_components_count": c,
        "reachability_score_percent": round(reachability_ratio * 100.0, 2),
        "topological_accuracy_score": round(accuracy_score, 2),
        "average_node_degree": round(avg_degree, 2)
    }
    return metrics

def export_graph_image(G, output_img_path="graph_preview.png"):
    fig, ax = plt.subplots(figsize=(10, 10), facecolor="#111111")
    ax.set_facecolor("#111111")

    for u, v in G.edges():
        if u in G.nodes and v in G.nodes:
            x1, y1 = G.nodes[u].get("x"), G.nodes[u].get("y")
            x2, y2 = G.nodes[v].get("x"), G.nodes[v].get("y")
            if None not in (x1, y1, x2, y2):
                ax.plot([x1, x2], [y1, y2], color="#3399FF", linewidth=0.8, alpha=0.7)

    xs = [data.get("x") for _, data in G.nodes(data=True) if data.get("x") is not None]
    ys = [data.get("y") for _, data in G.nodes(data=True) if data.get("y") is not None]
    if xs and ys:
        ax.scatter(xs, ys, s=2, color="#00FFAA", alpha=0.6)

    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_img_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close("all")
    print(f"Visual preview image saved to: {output_img_path}")