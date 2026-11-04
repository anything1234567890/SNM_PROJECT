import math
import cv2
import numpy as np
import networkx as nx
import requests
from PIL import Image
from io import BytesIO
from skimage.morphology import skeletonize

def fetch_satellite_image(bbox):
    south, north, west, east = bbox
    url = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export"
    params = {
        "bbox": f"{west},{south},{east},{north}",
        "bboxSR": "4326",
        "imageSR": "4326",
        "size": "1024,1024",
        "format": "png",
        "f": "image"
    }
    headers = {"User-Agent": "FloodSimSatelliteExtractor/1.0"}
    response = requests.get(url, params=params, headers=headers)
    response.raise_for_status()
    
    img = Image.open(BytesIO(response.content)).convert("RGB")
    return np.array(img)

def extract_road_skeleton(image_rgb):
    """
    Extracts continuous road corridors while filtering out roof textures and field boundaries.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    
    # 1. Blur to eliminate roof tile textures and small yard noise
    blurred = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    
    # 2. Otsu thresholding combined with morphology
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 3. Morphological opening to remove small disconnected building noise
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_small, iterations=1)
    
    # 4. Morphological closing with line kernels to join continuous road segments
    kernel_road = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    roads_joined = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_road, iterations=2)
    
    # 5. Skeletonize to 1-pixel centerlines
    bool_mask = roads_joined > 0
    skeleton = skeletonize(bool_mask)
    return skeleton.astype(np.uint8)

def skeleton_to_graph(skeleton, bbox):
    """
    Builds a topological graph and simplifies straight road runs into single edges.
    """
    south, north, west, east = bbox
    height, width = skeleton.shape
    
    G = nx.Graph()
    G.graph["crs"] = "EPSG:4326"
    
    y_indices, x_indices = np.where(skeleton > 0)
    
    def pixel_to_gps(px, py):
        lat = north - (py / height) * (north - south)
        lon = west + (px / width) * (east - west)
        return lat, lon

    pixel_set = set(zip(y_indices, x_indices))
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    
    for y, x in pixel_set:
        lat1, lon1 = pixel_to_gps(x, y)
        node_id1 = f"{lat1:.6f}_{lon1:.6f}"
        G.add_node(node_id1, x=lon1, y=lat1)
        
        for dy, dx in directions:
            ny, nx_pt = y + dy, x + dx
            if (ny, nx_pt) in pixel_set:
                lat2, lon2 = pixel_to_gps(nx_pt, ny)
                node_id2 = f"{lat2:.6f}_{lon2:.6f}"
                dist_m = math.hypot(dx, dy) * 111000 * ((north - south) / height)
                G.add_edge(node_id1, node_id2, length=dist_m)
                
    # Topological simplification: compress degree-2 intermediate nodes
    print("Compressing degree-2 intermediate nodes into simplified street corridors...")
    G_simplified = nx.Graph(G)
    nodes_to_remove = [n for n in G_simplified.nodes() if G_simplified.degree(n) == 2]
    
    for n in nodes_to_remove:
        if n in G_simplified and G_simplified.degree(n) == 2:
            neighbors = list(G_simplified.neighbors(n))
            u, v = neighbors[0], neighbors[1]
            if u != v:
                total_len = G_simplified[u][n].get("length", 1) + G_simplified[n][v].get("length", 1)
                G_simplified.add_edge(u, v, length=total_len)
                G_simplified.remove_node(n)
                
    # Remove tiny isolated noise components (under 5 nodes)
    for component in list(nx.connected_components(G_simplified)):
        if len(component) < 5:
            for node in component:
                G_simplified.remove_node(node)
                
    return G_simplified