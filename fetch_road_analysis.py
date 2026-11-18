
import os
import geopandas as gpd

# Read road segments and flood polygons
roads = gpd.read_file("outputs/roads.geojson")
floods = gpd.read_file("outputs/flood_extent.geojson")

# Use a projected CRS so lengths are measured in metres
roads = roads.to_crs("EPSG:32643")
floods = floods.to_crs(roads.crs)

# Combine flood polygons
flood_area = floods.geometry.union_all()

# Calculate how much of each road overlaps the flood area
roads["road_length"] = roads.geometry.length

roads["flooded_length"] = roads.geometry.intersection(
    flood_area
).length

roads["flood_exposure_percent"] = (
    roads["flooded_length"] /
    roads["road_length"].replace(0, float("nan"))
) * 100

# Keep roads with some flood overlap
vulnerable = roads[
    roads["flood_exposure_percent"] > 0
].copy()

os.makedirs("outputs", exist_ok=True)

vulnerable.to_file(
    "outputs/vulnerable_roads.geojson",
    driver="GeoJSON"
)

vulnerable.drop(columns="geometry").to_csv(
    "outputs/vulnerable_roads.csv",
    index=False
)

print(vulnerable[
    ["u", "v", "key", "flood_exposure_percent"]
].head(10))
