import folium
import geopandas as gpd
import networkx as nx

G = nx.read_graphml("output_graph.graphml")
flood_gdf = gpd.read_file("flood_stages.geojson")

# Center map on Begusarai graph
xs = [float(d["x"]) for _, d in G.nodes(data=True)]
ys = [float(d["y"]) for _, d in G.nodes(data=True)]
center_lat, center_lon = sum(ys) / len(ys), sum(xs) / len(xs)

m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="OpenStreetMap")

# Add Stage 0 (Actual River Channel in bright cyan) and Stage 3 (Floodplain in translucent blue)
stage_0 = flood_gdf[flood_gdf["stage"] == 0]
stage_3 = flood_gdf[flood_gdf["stage"] == 3]

folium.GeoJson(
    stage_3,
    name="Stage 3 Flood Extent",
    style_function=lambda x: {"fillColor": "#0066ff", "color": "#0044cc", "weight": 1, "fillOpacity": 0.25}
).add_to(m)

folium.GeoJson(
    stage_0,
    name="Extracted River Channel",
    style_function=lambda x: {"fillColor": "#00ffff", "color": "#ff0055", "weight": 2, "fillOpacity": 0.6}
).add_to(m)

# Overlay road edges
for u, v in G.edges():
    lat1, lon1 = float(G.nodes[u]["y"]), float(G.nodes[u]["x"])
    lat2, lon2 = float(G.nodes[v]["y"]), float(G.nodes[v]["x"])
    folium.PolyLine([(lat1, lon1), (lat2, lon2)], color="#222222", weight=1.2, opacity=0.7).add_to(m)

folium.LayerControl().add_to(m)
m.save("verify_map.html")
print("Saved verify_map.html — open it in your browser to inspect alignment against the real map!")