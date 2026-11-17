import random
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from flood_sim_engine import FloodSimulationEngine

app = FastAPI(title="Flood Resilience & Tactical Evacuation Command Center")

engine = FloodSimulationEngine(
    graphml_path="output_graph.graphml",
    flood_geojson="flood_stages.geojson"
)

# Shared movement constant: 3500m per flood stage (700m per tick across 5 ticks/stage)
METERS_PER_STAGE = 3500.0


class RouteRequest(BaseModel):
    start_node: str
    stage: int = 0


class BatchRequest(BaseModel):
    num_agents: int = 40


@app.get("/")
def serve_dashboard():
    return FileResponse("sim_dashboard.html")


@app.get("/api/graph_init")
def get_graph_init():
    nodes = []
    for n, d in engine.G.nodes(data=True):
        nodes.append({
            "id": n,
            "lat": d["y"],
            "lon": d["x"],
            "submerged_at": d["submerged_at_stage"],
            "is_haven": n in engine.safe_havens
        })

    edges = []
    for u, v, d in engine.G.edges(data=True):
        edges.append({
            "u": u,
            "v": v,
            "lat1": engine.G.nodes[u]["y"],
            "lon1": engine.G.nodes[u]["x"],
            "lat2": engine.G.nodes[v]["y"],
            "lon2": engine.G.nodes[v]["x"],
            "length": d["length"],
            "submerged_at": d["submerged_at_stage"]
        })

    flood_geojson = json.loads(engine.flood_gdf.to_json())

    xs = [n["lon"] for n in nodes]
    ys = [n["lat"] for n in nodes]
    center = [sum(ys) / len(ys), sum(xs) / len(xs)]

    return {
        "center": center,
        "nodes": nodes,
        "edges": edges,
        "flood_geojson": flood_geojson,
        "total_havens": len(engine.safe_havens),
        "meters_per_stage": METERS_PER_STAGE
    }


@app.get("/api/stage_metrics/{stage}")
def get_stage_metrics(stage: int):
    if stage < 0 or stage > 4:
        raise HTTPException(status_code=400, detail="Stage must be between 0 and 4")

    G_t = engine.get_graph_at_stage(stage)
    ebc = engine.stage_centrality.get(stage, {})

    sorted_ebc = sorted(ebc.items(), key=lambda x: x[1], reverse=True)[:10]
    top_bottlenecks = []
    for (u, v), score in sorted_ebc:
        if u in engine.G.nodes and v in engine.G.nodes:
            top_bottlenecks.append({
                "edge": f"{u[:9]} -> {v[:9]}",
                "u": u,
                "v": v,
                "lat1": engine.G.nodes[u]["y"],
                "lon1": engine.G.nodes[u]["x"],
                "lat2": engine.G.nodes[v]["y"],
                "lon2": engine.G.nodes[v]["x"],
                "centrality": round(score, 5)
            })

    total_edges = engine.G.number_of_edges()
    surviving_edges = G_t.number_of_edges()
    drowned_pct = round(((total_edges - surviving_edges) / max(total_edges, 1)) * 100, 1)

    return {
        "stage": stage,
        "surviving_nodes": G_t.number_of_nodes(),
        "surviving_edges": surviving_edges,
        "drowned_percent": drowned_pct,
        "top_bottlenecks": top_bottlenecks
    }


@app.post("/api/route")
def calculate_routes(req: RouteRequest):
    if req.start_node not in engine.G.nodes:
        raise HTTPException(status_code=404, detail="Node not found")

    resilient = engine.compute_safe_evacuation_path(req.start_node, req.stage, mode="resilient")
    naive = engine.compute_safe_evacuation_path(req.start_node, req.stage, mode="naive")

    def attach_coords(route_dict):
        coords = [[engine.G.nodes[n]["y"], engine.G.nodes[n]["x"]] for n in route_dict.get("path", [])]
        route_dict["coords"] = coords
        return route_dict

    return {
        "resilient": attach_coords(resilient),
        "naive": attach_coords(naive)
    }


def _simulate_single_agent(start_node: str, mode: str):
    """Simulates an agent walking across stages 0 -> 4 with exact distance parity."""
    curr = start_node

    for stage in range(5):
        if curr in engine.safe_havens:
            return True
        if engine.G.nodes[curr]["submerged_at_stage"] <= stage:
            return False

        plan = engine.compute_safe_evacuation_path(curr, stage, mode=mode)
        if plan["status"] != "ROUTING" or len(plan["path"]) < 2:
            return False

        budget = METERS_PER_STAGE
        path = plan["path"]
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if engine.G[u][v]["submerged_at_stage"] <= stage or engine.G.nodes[v]["submerged_at_stage"] <= stage:
                break
            seg_len = engine.G[u][v]["length"]
            if budget >= seg_len:
                budget -= seg_len
                curr = v
                if curr in engine.safe_havens:
                    return True
            else:
                break

    return curr in engine.safe_havens


@app.post("/api/batch_simulate")
def run_batch_simulation(req: BatchRequest):
    at_risk_pool = [
        n for n, d in engine.G.nodes(data=True)
        if 1 <= d["submerged_at_stage"] <= 4 and n not in engine.safe_havens
    ]
    if not at_risk_pool:
        at_risk_pool = [n for n in engine.G.nodes() if n not in engine.safe_havens]

    sample_size = min(req.num_agents, len(at_risk_pool))
    spawn_nodes = random.sample(at_risk_pool, sample_size)

    resilient_survived = 0
    naive_survived = 0
    trials = []

    for node_id in spawn_nodes:
        r_ok = _simulate_single_agent(node_id, mode="resilient")
        n_ok = _simulate_single_agent(node_id, mode="naive")
        if r_ok:
            resilient_survived += 1
        if n_ok:
            naive_survived += 1
        trials.append({
            "node": node_id,
            "lat": engine.G.nodes[node_id]["y"],
            "lon": engine.G.nodes[node_id]["x"],
            "resilient_survived": r_ok,
            "naive_survived": n_ok
        })

    return {
        "total_agents": sample_size,
        "resilient_survived": resilient_survived,
        "naive_survived": naive_survived,
        "resilient_accuracy": round((resilient_survived / sample_size) * 100, 1),
        "naive_accuracy": round((naive_survived / sample_size) * 100, 1),
        "trials": trials
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("sim_app:app", host="127.0.0.1", port=8000, reload=True)