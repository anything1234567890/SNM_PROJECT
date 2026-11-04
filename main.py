import os
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from graph_builder import search_india_places, build_graph_for_place

app = FastAPI(title="India Region Graph Builder for Flood Simulation")

# Ensure static preview directory exists
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/api/autocomplete")
def autocomplete(q: str = Query(..., min_length=2)):
    """
    Returns search suggestions strictly restricted to India.
    """
    try:
        results = search_india_places(q, limit=5)
        suggestions = []
        for item in results:
            suggestions.append({
                "display_name": item["display_name"],
                "data": item
            })
        return {"suggestions": suggestions}
    except Exception as e:
        return {"error": str(e), "suggestions": []}

@app.post("/api/generate-graph")
def generate_graph(place_data: dict):
    """
    Builds the graph, creates the visualization image, and calculates accuracy metrics.
    """
    try:
        output_graphml = "static/output_graph.graphml"
        output_png = "static/graph_preview.png"
        
        G, metrics = build_graph_for_place(
            place_data, 
            output_path=output_graphml, 
            preview_img=output_png
        )
        
        return {
            "status": "success",
            "metrics": metrics,
            "preview_url": f"/static/graph_preview.png?t={os.path.getmtime(output_png)}",
            "graphml_url": "/static/output_graph.graphml"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/download-graphml")
def download_graphml():
    file_path = "static/output_graph.graphml"
    if os.path.exists(file_path):
        return FileResponse(file_path, filename="region_flood_graph.graphml")
    return {"error": "File not found"}