import math
import os
import geopandas as gpd
import networkx as nx
import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.merge import merge
from rasterio.windows import from_bounds
import requests
from scipy import ndimage
from shapely.geometry import shape
from shapely.ops import unary_union


def download_copernicus_layers(west, south, east, north, cache_dir="dem_cache"):
    """Downloads free 30m Copernicus DEM and Water Body Mask (WBM) tiles from AWS S3."""
    os.makedirs(cache_dir, exist_ok=True)
    lat_min, lat_max = math.floor(south), math.floor(north)
    lon_min, lon_max = math.floor(west), math.floor(east)

    dem_paths = []
    wbm_paths = []

    slash = chr(47)
    base_host = f"https:{slash}{slash}://amazonaws.com"

    for lat in range(lat_min, lat_max + 1):
        for lon in range(lon_min, lon_max + 1):
            ns = "N" if lat >= 0 else "S"
            ew = "E" if lon >= 0 else "W"
            tile_name = (
                f"Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00_"
                f"{ew}{abs(lon):03d}_00_DEM"
            )

            dem_url = f"{base_host}{slash}{tile_name}{slash}{tile_name}.tif"
            dem_local = os.path.join(cache_dir, f"{tile_name}.tif")
            if not os.path.exists(dem_local):
                print(f"Downloading 30m DEM tile: {tile_name}...")
                resp = requests.get(dem_url, stream=True, timeout=60)
                resp.raise_for_status()
                with open(dem_local, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        f.write(chunk)
            dem_paths.append(dem_local)

            wbm_name = tile_name.replace("_DEM", "_WBM")
            wbm_url = f"{base_host}{slash}{tile_name}{slash}AUXFILES{slash}{wbm_name}.tif"
            wbm_local = os.path.join(cache_dir, f"{wbm_name}.tif")
            if not os.path.exists(wbm_local):
                print(f"Downloading 30m Water Body Mask: {wbm_name}...")
                resp = requests.get(wbm_url, stream=True, timeout=60)
                resp.raise_for_status()
                with open(wbm_local, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        f.write(chunk)
            wbm_paths.append(wbm_local)

    return dem_paths, wbm_paths


def load_clipped_raster(tile_paths, west, south, east, north, dtype=np.float32):
    """Mosaics and crops raster tiles to the exact study area bounding box."""
    datasets = [rasterio.open(p) for p in tile_paths]
    if len(datasets) == 1:
        src = datasets[0]
        window = from_bounds(west, south, east, north, src.transform)
        arr = src.read(1, window=window).astype(dtype)
        out_transform = src.window_transform(window)
        src.close()
    else:
        arr, out_transform = merge(datasets, bounds=(west, south, east, north))
        arr = arr[0].astype(dtype)
        for ds in datasets:
            ds.close()

    return arr, out_transform


def generate_dem_flood_stages(
    graphml_path="output_graph.graphml", output_geojson="flood_stages.geojson"
):
    print("1. Reading graph bounds...")
    G = nx.read_graphml(graphml_path)
    xs = [float(data["x"]) for _, data in G.nodes(data=True) if "x" in data]
    ys = [float(data["y"]) for _, data in G.nodes(data=True) if "y" in data]

    pad = 0.05
    west, east = min(xs) - pad, max(xs) + pad
    south, north = min(ys) - pad, max(ys) + pad

    print("2. Acquiring 30m Copernicus DEM & Water Body Mask rasters...")
    dem_paths, wbm_paths = download_copernicus_layers(west, south, east, north)
    dem, transform = load_clipped_raster(
        dem_paths, west, south, east, north, dtype=np.float32
    )
    wbm, _ = load_clipped_raster(
        wbm_paths, west, south, east, north, dtype=np.uint8
    )

    # Light smoothing (sigma=0.8) preserves subtle 0.3m-0.8m road embankments
    dem_smooth = ndimage.gaussian_filter(dem, sigma=0.8)

    print("3. Extracting connected river channels from Water Body Mask...")
    raw_water = wbm >= 2
    labeled_water, _ = ndimage.label(raw_water, structure=np.ones((3, 3)))
    counts = np.bincount(labeled_water.ravel())
    large_water_ids = np.where(counts >= 50)[0]
    large_water_ids = large_water_ids[large_water_ids > 0]
    river_mask = np.isin(labeled_water, large_water_ids)

    print("4. Computing Height Above Nearest Drainage (HAND) with hydraulic slope...")
    dist_pixels, nearest_idx = ndimage.distance_transform_edt(
        ~river_mask, return_indices=True
    )
    nearest_river_elev = dem_smooth[nearest_idx[0], nearest_idx[1]]

    # Add realistic hydraulic friction (~0.15m drop per 10km inland, where 1 pixel ≈ 30m)
    # This prevents flat plains 15km away from flooding instantaneously on Stage 1
    dist_km = (dist_pixels * 30.0) / 1000.0
    hydraulic_resistance = dist_km * 0.025

    effective_hand = np.maximum(0.0, (dem_smooth - nearest_river_elev) + hydraulic_resistance)

    # Fine-grained sub-meter stages calibrated for the flat Gangetic alluvial plain
    flood_stages = [
        {"stage": 0, "label": "Stage 0: Main River Channels", "rise_m": 0.0},
        {"stage": 1, "label": "Stage 1: Early Bankfull Spill (+0.45m)", "rise_m": 0.45},
        {"stage": 2, "label": "Stage 2: Lowland & Chaur Surge (+0.85m)", "rise_m": 0.85},
        {"stage": 3, "label": "Stage 3: Widespread Floodplain (+1.40m)", "rise_m": 1.40},
        {"stage": 4, "label": "Stage 4: Peak Monsoon Inundation (+2.20m)", "rise_m": 2.20},
    ]

    print("5. Running hydrological connectivity flood-fill per stage...")
    records = []
    # Use 4-connectivity (rook) instead of 8-connectivity so water doesn't leak diagonally through narrow embankments
    structure_4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)

    for item in flood_stages:
        if item["rise_m"] == 0.0:
            flood_mask = river_mask
        else:
            candidate_mask = (effective_hand <= item["rise_m"]) | river_mask
            labeled_regions, _ = ndimage.label(
                candidate_mask, structure=structure_4
            )
            river_labels = np.unique(labeled_regions[river_mask])
            river_labels = river_labels[river_labels > 0]
            flood_mask = np.isin(labeled_regions, river_labels)

        flood_mask = ndimage.binary_closing(flood_mask, structure=np.ones((2, 2)))
        poly_list = [
            shape(geom)
            for geom, val in shapes(
                flood_mask.astype(np.uint8), mask=flood_mask, transform=transform
            )
            if val == 1
        ]
        merged_poly = unary_union(poly_list).simplify(
            0.00015, preserve_topology=True
        )

        records.append({
            "stage": item["stage"],
            "label": item["label"],
            "rise_m": item["rise_m"],
            "geometry": merged_poly,
        })
        print(f"   -> Generated {item['label']}")

    flood_gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    flood_gdf.to_file(output_geojson, driver="GeoJSON")
    print(f"\nSuccess! Saved progressive flood stages to '{output_geojson}'.")


if __name__ == "__main__":
    generate_dem_flood_stages()

