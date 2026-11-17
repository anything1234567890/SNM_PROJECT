import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString, Point
from shapely.prepared import prep


class FloodSimulationEngine:
    def __init__(self, graphml_path="output_graph.graphml", flood_geojson="flood_stages.geojson"):
        print("Initializing Flood Simulation Engine...")
        self.base_graph = nx.read_graphml(graphml_path)
        self.flood_gdf = gpd.read_file(flood_geojson).sort_values("stage")

        self.G = self._prepare_base_graph(self.base_graph)
        self._precompute_inundation_schedule()

        # Select top ~40 well-connected high-ground hubs as Relief Shelters (Safe Havens)
        dry_candidates = [
            (n, self.G.degree(n)) for n, d in self.G.nodes(data=True)
            if d.get("submerged_at_stage", 99) > 4
        ]
        dry_candidates.sort(key=lambda x: x[1], reverse=True)
        selected_havens = []
        for node_id, _ in dry_candidates:
            nx_coord, ny_coord = self.G.nodes[node_id]["x"], self.G.nodes[node_id]["y"]
            too_close = any(
                ((nx_coord - self.G.nodes[h]["x"]) ** 2 + (ny_coord - self.G.nodes[h]["y"]) ** 2) ** 0.5 < 0.022
                for h in selected_havens
            )
            if not too_close:
                selected_havens.append(node_id)
            if len(selected_havens) >= 40:
                break

        self.safe_havens = set(selected_havens)
        print(f"Designated {len(self.safe_havens)} major high-ground Relief Shelters.")

        self.stage_centrality = {}
        self._precompute_centralities()

    def _prepare_base_graph(self, G_raw):
        G = nx.Graph()
        for n, data in G_raw.nodes(data=True):
            G.add_node(
                n,
                x=float(data["x"]),
                y=float(data["y"]),
                submerged_at_stage=99
            )
        for u, v, data in G_raw.edges(data=True):
            if u == v:
                continue
            length = float(data.get("length", 100.0))
            if G.has_edge(u, v):
                if length < G[u][v]["length"]:
                    G[u][v]["length"] = length
            else:
                G.add_edge(u, v, length=length, submerged_at_stage=99)
        return G

    def _precompute_inundation_schedule(self):
        print("Mapping flood stages onto road nodes and edges...")
        stage_polys = []
        for _, row in self.flood_gdf.iterrows():
            stage_polys.append((int(row["stage"]), prep(row.geometry)))

        for n, data in self.G.nodes(data=True):
            pt = Point(data["x"], data["y"])
            for stage_num, prepared_poly in stage_polys:
                if prepared_poly.contains(pt):
                    data["submerged_at_stage"] = stage_num
                    break

        for u, v, data in self.G.edges(data=True):
            u_stage = self.G.nodes[u]["submerged_at_stage"]
            v_stage = self.G.nodes[v]["submerged_at_stage"]
            earliest = min(u_stage, v_stage)

            line = LineString([
                (self.G.nodes[u]["x"], self.G.nodes[u]["y"]),
                (self.G.nodes[v]["x"], self.G.nodes[v]["y"])
            ])
            for stage_num, prepared_poly in stage_polys:
                if stage_num >= earliest:
                    break
                if prepared_poly.intersects(line):
                    earliest = stage_num
                    break
            data["submerged_at_stage"] = earliest

    def _precompute_centralities(self):
        print("Precomputing dynamic Edge Betweenness Centrality across flood stages...")
        for stage in range(5):
            G_active = self.get_graph_at_stage(stage)
            k_pivots = min(250, G_active.number_of_nodes())
            ebc = nx.edge_betweenness_centrality(G_active, k=k_pivots, weight="length", normalized=True)
            self.stage_centrality[stage] = ebc

    def get_graph_at_stage(self, stage: int):
        G_t = nx.Graph()
        for n, data in self.G.nodes(data=True):
            if data["submerged_at_stage"] > stage:
                G_t.add_node(n, **data)
        for u, v, data in self.G.edges(data=True):
            if data["submerged_at_stage"] > stage and u in G_t and v in G_t:
                G_t.add_edge(u, v, **data)
        return G_t

    def compute_safe_evacuation_path(self, start_node: str, current_stage: int, mode: str = "resilient"):
        G_t = self.get_graph_at_stage(current_stage)
        if start_node not in G_t:
            return {"status": "SUBMERGED", "path": [], "distance_m": 0}

        if start_node in self.safe_havens:
            return {"status": "SAFE", "path": [start_node], "distance_m": 0}

        reachable_nodes = nx.node_connected_component(G_t, start_node)
        candidate_havens = [h for h in self.safe_havens if h in reachable_nodes]

        if not candidate_havens:
            return {"status": "TRAPPED", "path": [], "distance_m": 0}

        def edge_weight(u, v, data):
            base_len = data["length"]
            if mode == "naive":
                return base_len

            drown_stage = min(
                data["submerged_at_stage"],
                G_t.nodes[u]["submerged_at_stage"],
                G_t.nodes[v]["submerged_at_stage"]
            )
            stages_left = drown_stage - current_stage

            # Full predictive lookahead across all remaining flood stages (1..4)
            if stages_left <= 1:
                flood_risk_mult = 12.0   # Imminent submersion on next stage tick
            elif stages_left == 2:
                flood_risk_mult = 5.0    # High mid-evacuation submersion risk
            elif stages_left == 3:
                flood_risk_mult = 2.2    # Moderate late-stage risk
            elif stages_left == 4:
                flood_risk_mult = 1.4    # Slight preference for permanently dry high ground
            else:
                flood_risk_mult = 1.0    # Permanently dry above Stage 4

            # Penalize dead-end / single-exit bottlenecks that can isolate the walker
            min_deg = min(G_t.degree(u), G_t.degree(v))
            redundancy_mult = 2.0 if min_deg <= 2 else 1.0

            return base_len * flood_risk_mult * redundancy_mult

        _, paths = nx.multi_source_dijkstra(G_t, sources=candidate_havens, weight=edge_weight)

        if start_node not in paths:
            return {"status": "TRAPPED", "path": [], "distance_m": 0}

        best_path = list(reversed(paths[start_node]))
        phys_dist = sum(G_t[best_path[i]][best_path[i + 1]]["length"] for i in range(len(best_path) - 1))

        return {
            "status": "ROUTING",
            "path": best_path,
            "target_haven": best_path[-1],
            "distance_m": round(phys_dist, 1)
        }