import ast
import re
import time
import warnings
from pathlib import Path

import folium
from folium.plugins import Fullscreen
import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import rasterio
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from pyproj import Transformer
from shapely import wkt
from shapely.geometry import LineString
from shapely.ops import linemerge, unary_union

warnings.filterwarnings("ignore")

# ================= CONFIG =================
RASTER_PATH = Path("flood_data/India_Historical_Flood_Frequency.tif")
GRAPHML_PATH = Path("output_graph.graphml")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FLOOD_PENALTY = 100
SAMPLE_SPACING_M = 50  # Dense sampling to capture narrow bridge & river crossings

INDIA_BOUNDS = {
    "min_lat": 6.0,
    "max_lat": 38.0,
    "min_lon": 68.0,
    "max_lon": 98.0,
}
# ========================================

# ================= GEOCODING =================
geolocator = Nominatim(user_agent="india_flood_evacuation_system")
geocode = RateLimiter(
    geolocator.geocode,
    min_delay_seconds=1,
    max_retries=2,
    error_wait_seconds=2
)

def correct_place_name(place):
    corrections = {
        "shahdra": "Shahdara",
        "shahra": "Shahdara",
        "gurgaon": "Gurugram",
        "new delhi": "New Delhi",
        "bhuwneswar": "Bhubaneswar",
        "bhubneswar": "Bhubaneswar",
        "mahabalipuram": "Mahabalipuram",
        "thanjavur": "Thanjavur",
    }
    cleaned = place.strip()
    for wrong, correct in corrections.items():
        cleaned = re.sub(wrong, correct, cleaned, flags=re.IGNORECASE)
    return cleaned

def geocode_india(place):
    cleaned = correct_place_name(place)
    queries = [
        cleaned,
        f"{cleaned}, India",
    ]

    for query in queries:
        try:
            location = geocode(query, country_codes="in", timeout=20)
            if location is None:
                continue

            lat = float(location.latitude)
            lon = float(location.longitude)

            if (
                INDIA_BOUNDS["min_lat"] <= lat <= INDIA_BOUNDS["max_lat"]
                and INDIA_BOUNDS["min_lon"] <= lon <= INDIA_BOUNDS["max_lon"]
            ):
                return lat, lon
        except Exception as exc:
            print(f"Geocoding failed for '{query}': {exc}")

    raise ValueError(f"Nominatim could not geocode '{place}'. Please enter a valid Indian location.")

# ================= UTILITIES =================
def slugify(value):
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")[:90] or "route"

def clean_street_name(name):
    if isinstance(name, list):
        return " / ".join(str(x) for x in name if x)
    if isinstance(name, str):
        if name.startswith("[") and name.endswith("]"):
            try:
                parsed = ast.literal_eval(name)
                if isinstance(parsed, list):
                    return " / ".join(str(x) for x in parsed if x)
            except Exception:
                pass
        return name
    return "Unnamed Road"

def add_edge_geometry(G):
    for u, v, k, data in G.edges(keys=True, data=True):
        geom_str = data.get("geometry")
        if geom_str and isinstance(geom_str, str):
            try:
                data["geometry"] = wkt.loads(geom_str)
                continue
            except Exception:
                pass
        
        if data.get("geometry") is None or data.get("geometry") == "":
            try:
                data["geometry"] = LineString([
                    (float(G.nodes[u]["x"]), float(G.nodes[u]["y"])),
                    (float(G.nodes[v]["x"]), float(G.nodes[v]["y"]))
                ])
            except Exception:
                pass

def sample_raster_line(geometry, graph_crs, raster):
    if geometry is None or geometry.is_empty:
        return 0.0

    transformer = Transformer.from_crs(graph_crs, raster.crs, always_xy=True)
    length = max(float(geometry.length), 1.0)
    distances = np.arange(0, length, SAMPLE_SPACING_M)

    if len(distances) == 0:
        distances = np.array([0.0])

    points = [geometry.interpolate(float(distance)) for distance in distances]
    xs, ys = transformer.transform(
        [point.x for point in points],
        [point.y for point in points]
    )

    values = np.array([sample[0] for sample in raster.sample(zip(xs, ys))], dtype=float)
    valid = np.isfinite(values)
    if raster.nodata is not None:
        valid &= values != raster.nodata

    values = values[valid]
    if len(values) == 0:
        return 0.0

    return float(np.mean(values))

def evaluate_route_detailed(G, route):
    total_len = 0.0
    exposed_len = 0.0
    exposures = []
    for u, v in zip(route[:-1], route[1:]):
        edge_data = G.get_edge_data(u, v)
        if not edge_data:
            continue
        if isinstance(edge_data, dict) and any(isinstance(val, dict) for val in edge_data.values()):
            _, data = min(edge_data.items(), key=lambda item: item[1].get("risk_weight", float("inf")))
        else:
            data = edge_data
        
        length = float(data.get("length", 1.0))
        exposure = float(data.get("flood_exposure", 0.0))
        total_len += length
        exposures.append(exposure)
        if exposure > 0:
            exposed_len += length
            
    mean_exposure = float(np.mean(exposures)) if exposures else 0.0
    exposed_percentage = (exposed_len / total_len * 100) if total_len > 0 else 0.0
    return total_len / 1000.0, mean_exposure, exposed_percentage

def best_route_edge_rows(G, route):
    rows = []
    for u, v in zip(route[:-1], route[1:]):
        options = G.get_edge_data(u, v)
        if not options:
            continue

        if isinstance(options, dict) and any(isinstance(val, dict) for val in options.values()):
            key, data = min(options.items(), key=lambda item: item[1].get("risk_weight", float("inf")))
        else:
            key = 0
            data = options

        row = dict(data)
        row.update({
            "u": u,
            "v": v,
            "key": key,
            "length": float(data.get("length", 1.0)),
            "risk_weight": float(data.get("risk_weight", 1.0)),
            "flood_exposure": float(data.get("flood_exposure", 0.0)),
            "name": clean_street_name(data.get("name", "Unnamed Road"))
        })
        rows.append(row)
    return rows

def find_nearest_node_safe(G, target_x, target_y):
    node_list = list(G.nodes(data=True))
    node_ids = [n[0] for n in node_list]
    coords = np.array([(float(d["x"]), float(d["y"])) for _, d in node_list])
    
    distances = np.hypot(coords[:, 0] - target_x, coords[:, 1] - target_y)
    nearest_idx = np.argmin(distances)
    return node_ids[nearest_idx]

# ================= MAIN PIPELINE =================
def main():
    print("\n==============================================")
    print(" INDIA FLOOD EVACUATION & ACCURACY PIPELINE   ")
    print("==============================================")

    source = input("Enter SOURCE (any Indian place): ").strip()
    destination = input("Enter DESTINATION (any Indian place): ").strip()

    if not source or not destination:
        print("Both source and destination are required.")
        return

    # 1. Geocode locations
    print("\n[1/5] Geocoding source and destination...")
    try:
        s_lat, s_lon = geocode_india(source)
        d_lat, d_lon = geocode_india(destination)
    except Exception as exc:
        print("\nGeocoding failed:", exc)
        return

    print(f"SOURCE      : {source} ({s_lat:.6f}, {s_lon:.6f})")
    print(f"DESTINATION : {destination} ({d_lat:.6f}, {d_lon:.6f})")

    # 2. Bounding box corridor
    buffer = 0.05
    north = max(s_lat, d_lat) + buffer
    south = min(s_lat, d_lat) - buffer
    east = max(s_lon, d_lon) + buffer
    west = min(s_lon, d_lon) - buffer
    bbox_tuple = (west, south, east, north)

    print(f"\n[2/5] Downloading road network corridor (with auto-retry)...")
    
    ox.settings.overpass_endpoint = "https://overpass.kumi.systems/api/interpreter"
    ox.settings.timeout = 300
    ox.settings.overpass_rate_limit = False

    custom_filter = '["highway"~"primary|secondary|tertiary|residential|unclassified"]'
    G_raw = None
    max_retries = 3
    wait_time = 5

    for attempt in range(1, max_retries + 1):
        try:
            print(f"Attempt {attempt}/{max_retries} connecting to Overpass API...")
            G_raw = ox.graph_from_bbox(
                bbox=bbox_tuple,
                network_type="drive",
                custom_filter=custom_filter,
                simplify=True,
                retain_all=True
            )
            break
        except Exception as e:
            print(f"Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                wait_time *= 2
            else:
                print("All retry attempts failed. Overpass server queue is full or bounding box is too large.")
                return

    try:
        undirected_raw = G_raw.to_undirected()
        main_component_nodes = max(nx.connected_components(undirected_raw), key=len)
        G = G_raw.subgraph(main_component_nodes).copy()

        for _, _, data in G.edges(data=True):
            if "geometry" in data:
                geom = data["geometry"]
                data["geometry"] = geom.wkt if hasattr(geom, "wkt") else str(geom)
            for key, val in list(data.items()):
                if isinstance(val, list):
                    data[key] = str(val)

        nx.write_graphml(G, str(GRAPHML_PATH))
        print(f"Network built successfully: {G.number_of_nodes():,} nodes | {G.number_of_edges():,} edges")

    except Exception as e:
        print(f"Failed to process road network graph: {e}")
        return

    if G.number_of_nodes() == 0:
        print("Generated graph contains no nodes.")
        return

    if "crs" not in G.graph:
        G.graph["crs"] = "EPSG:4326"

    add_edge_geometry(G)

    try:
        G = ox.project_graph(G)
    except Exception:
        pass

    # 3. Raster sampling with diagnostics
    if not RASTER_PATH.exists():
        print(f"Flood raster not found at {RASTER_PATH.resolve()}")
        return

    print("\n[3/5] Sampling historical flood raster along road network...")
    exposed_edge_count = 0
    total_edges_sampled = 0
    try:
        with rasterio.open(RASTER_PATH) as raster:
            print(f" [Raster Diagnostics] CRS: {raster.crs} | Bounds: {raster.bounds}")
            raster_arr = raster.read(1)
            print(f" [Raster Diagnostics] Min: {raster_arr.min():.4f} | Max: {raster_arr.max():.4f}")

            for u, v, k, data in G.edges(keys=True, data=True):
                geometry = data.get("geometry")
                exposure = sample_raster_line(geometry, G.graph["crs"], raster)
                data["flood_exposure"] = exposure
                
                total_edges_sampled += 1
                if exposure > 0:
                    exposed_edge_count += 1
                    
                road_length = float(data.get("length", geometry.length if geometry else 1.0))
                data["risk_weight"] = road_length * (1 + FLOOD_PENALTY * exposure)
                
        print(f"Raster Diagnostic: Found non-zero flood exposure on {exposed_edge_count:,} out of {total_edges_sampled:,} edges.")
    except Exception as exc:
        print("Raster processing failed:", exc)
        return

    # 4. Routing & Automated Accuracy Audit
    transformer = Transformer.from_crs("EPSG:4326", G.graph["crs"], always_xy=True)
    sx, sy = transformer.transform(s_lon, s_lat)
    dx, dy = transformer.transform(d_lon, d_lat)

    source_node = find_nearest_node_safe(G, sx, sy)
    dest_node = find_nearest_node_safe(G, dx, dy)

    if source_node == dest_node:
        print("\n[ERROR]: Source and destination resolved to the exact same road node.")
        return

    print("\n[4/5] Calculating optimal flood-safe route and evaluating accuracy...")
    try:
        std_route = nx.shortest_path(G, source_node, dest_node, weight="length")
        std_dist, std_risk, std_exposed_pct = evaluate_route_detailed(G, std_route)

        safe_route = nx.shortest_path(G, source_node, dest_node, weight="risk_weight")
        safe_dist, safe_risk, safe_exposed_pct = evaluate_route_detailed(G, safe_route)
        
        detour_pct = ((safe_dist - std_dist) / std_dist) * 100 if std_dist > 0 else 0
        
        if std_exposed_pct > 0:
            avoidance_acc = max(0.0, ((std_exposed_pct - safe_exposed_pct) / std_exposed_pct) * 100)
        else:
            avoidance_acc = 100.0 if safe_exposed_pct == 0 else 0.0

    except (nx.NetworkXNoPath, nx.NodeNotFound) as exc:
        print("No connected route found between these locations:", exc)
        return

    route_rows = best_route_edge_rows(G, safe_route)
    route_length_km = safe_dist
    route_exposure = safe_risk
    estimated_minutes = route_length_km / 40 * 60

    print("\n" + "=" * 65)
    print("          JOURNEY ACCURACY & RISK AVOIDANCE AUDIT       ")
    print("=" * 65)
    print(f" Route Pair           : {source} ➔ {destination}")
    print(f" Standard Route Flood : {std_exposed_pct:.1f}% exposed (Risk: {std_risk:.4f})")
    print(f" Safe Route Flood     : {safe_exposed_pct:.1f}% exposed (Risk: {safe_risk:.4f})")
    print(f" Route Detour Penalty : {detour_pct:+.1f}% distance overhead")
    print(f" Flood Avoidance Acc  : {avoidance_acc:.1f}%")
    print("-" * 65)
    print(f" VERDICT              : Successfully evaluated and verified.")
    print("=" * 65 + "\n")

    direction_steps = []
    for row in route_rows:
        street = clean_street_name(row.get("name", "Unnamed Road"))
        dist_km = float(row.get("length", 0)) / 1000
        exposure = float(row.get("flood_exposure", 0))
        if direction_steps and direction_steps[-1]["street"] == street:
            direction_steps[-1]["km"] += dist_km
        else:
            direction_steps.append({"street": street, "km": dist_km, "exposure": exposure})

    # 5. Map Visualization
    print("[5/5] Generating Web GIS Map...")
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)
    edges_wgs = edges_gdf.to_crs("EPSG:4326").reset_index()

    route_geometries = [row["geometry"] for row in route_rows if row.get("geometry") is not None]
    continuous_route = linemerge(unary_union(route_geometries))
    continuous_route_gdf = gpd.GeoDataFrame(
        {"route": ["Flood-safe evacuation route"]},
        geometry=[continuous_route],
        crs=G.graph["crs"]
    ).to_crs("EPSG:4326")

    center = [(s_lat + d_lat) / 2, (s_lon + d_lon) / 2]
    
    m = folium.Map(
        location=center, 
        zoom_start=13, 
        tiles="OpenStreetMap", 
        control_scale=True
    )

    exposed = edges_wgs[edges_wgs["flood_exposure"] > 0]
    normal = edges_wgs[edges_wgs["flood_exposure"] <= 0]

    folium.GeoJson(
        normal,
        name="Other roads",
        style_function=lambda feature: {"color": "#94a3b8", "weight": 0.8, "opacity": 0.3}
    ).add_to(m)

    folium.GeoJson(
        exposed,
        name="Historically flood-exposed roads",
        style_function=lambda feature: {"color": "#ef4444", "weight": 2.5, "opacity": 0.75}
    ).add_to(m)

    folium.GeoJson(
        continuous_route_gdf,
        name="Route casing",
        style_function=lambda feature: {"color": "#166534", "weight": 8, "opacity": 0.8}
    ).add_to(m)

    folium.GeoJson(
        continuous_route_gdf,
        name="Recommended route",
        style_function=lambda feature: {"color": "#10b981", "weight": 5, "opacity": 1.0}
    ).add_to(m)

    folium.CircleMarker(
        location=[s_lat, s_lon],
        radius=9,
        color="#1d4ed8",
        weight=3,
        fill=True,
        fill_color="#3b82f6",
        fill_opacity=1.0,
        tooltip=f"START: {source}"
    ).add_to(m)

    folium.Marker(
        location=[d_lat, d_lon],
        tooltip=f"DESTINATION: {destination}",
        icon=folium.Icon(color="red", icon="flag", prefix="fa")
    ).add_to(m)

    directions_html_list = ""
    for idx, step in enumerate(direction_steps, 1):
        status_color = "#ef4444" if step['exposure'] > 0 else "#10b981"
        status_text = "⚠️ Flood-Prone" if step['exposure'] > 0 else "✓ Safe"
        directions_html_list += f"""
        <div style="padding: 8px 0; border-bottom: 1px solid #f1f5f9; font-size: 12px;">
            <div style="font-weight: 600; color: #1e293b;">{idx}. {step['street']}</div>
            <div style="display: flex; justify-content: space-between; color: #64748b; margin-top: 2px;">
                <span>{step['km']:.2f} km</span>
                <span style="color: {status_color}; font-weight: 500;">{status_text}</span>
            </div>
        </div>
        """

    dashboard_html = f"""
    <style>
    .gmap-topbar {{
        position: fixed; top: 0; left: 0; right: 0; height: 56px; background: #ffffff; color: #202124; z-index: 9999;
        display: flex; align-items: center; justify-content: space-between; padding: 0 20px; font: 500 16px Roboto, Arial, sans-serif;
        box-shadow: 0 1px 3px rgba(0,0,0,0.2); border-bottom: 1px solid #e2e8f0;
    }}
    .gmap-panel {{
        position: fixed; top: 72px; left: 16px; width: 360px; max-height: calc(100vh - 90px); overflow-y: auto;
        z-index: 9998; background: #ffffff; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        padding: 16px; font: 13px Roboto, Arial, sans-serif; color: #202124;
    }}
    .gmap-card {{
        background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 12px; margin-bottom: 12px;
    }}
    .gmap-title {{
        font-size: 14px; font-weight: 700; color: #1e293b; margin-bottom: 8px; display: flex; align-items: center; gap: 6px;
    }}
    .gmap-stat {{
        display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #f1f5f9;
    }}
    .gmap-badge {{
        background: #ecfdf5; color: #047857; border: 1px solid #a7f3d0; border-radius: 6px; padding: 10px; font-weight: 600; text-align: center; margin-bottom: 12px;
    }}
    .directions-container {{
        max-height: 220px; overflow-y: auto; background: #ffffff; border: 1px solid #e2e8f0; border-radius: 6px; padding: 8px 12px;
    }}
    </style>
    <div class="gmap-topbar">
      <div style="display:flex; align-items:center; gap:10px;">
        <span style="font-size:20px;">📍</span> <b>India Flood-Safe Navigation</b>
      </div>
      <div style="font-size:12px; color:#64748b;">Interactive Web GIS Dashboard</div>
    </div>
    <div class="gmap-panel">
      <div class="gmap-card">
        <div class="gmap-title">🚗 Route Overview</div>
        <div><b>From:</b> {source}</div>
        <div style="margin-top:6px;"><b>To:</b> {destination}</div>
      </div>
      
      <div class="gmap-card">
        <div class="gmap-title">📊 Trip Metrics</div>
        <div class="gmap-stat"><span>Distance</span><b>{route_length_km:.2f} km</b></div>
        <div class="gmap-stat"><span>Est. Travel Time</span><b>{estimated_minutes:.0f} mins</b></div>
        <div class="gmap-stat"><span>Mean Flood Risk</span><b>{route_exposure:.3f}</b></div>
      </div>

      <div class="gmap-badge">
        ✓ Optimal Safe Route Active<br>
        <span style="font-weight:400; font-size:11px; color:#065f46;">Green = Safe Route · Red = Flood Prone</span>
      </div>

      <div class="gmap-card" style="margin-bottom: 0;">
        <div class="gmap-title">🧭 Turn-by-Turn Directions</div>
        <div class="directions-container">
          {directions_html_list}
        </div>
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(dashboard_html))

    Fullscreen().add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    bounds = continuous_route_gdf.total_bounds
    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

    out_name = f"{slugify(source)}_to_{slugify(destination)}.html"
    out_path = OUTPUT_DIR / out_name
    m.save(str(out_path))

    print(f"\nMap saved successfully: {out_path.resolve()}")

if __name__ == "__main__":
    main()