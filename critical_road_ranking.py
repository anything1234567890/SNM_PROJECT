
import os
import csv
import networkx as nx
import pandas as pd

GRAPH_PATH = "output_graph.graphml"
OUTPUT_PATH = "outputs/critical_road_ranking.csv"

SAMPLE_SIZE = 500
CENTRALITY_WEIGHT = 0.5
FLOOD_WEIGHT = 0.5


def normalize(values):
    values = pd.Series(values, dtype=float)

    minimum = values.min()
    maximum = values.max()

    if maximum == minimum:
        return pd.Series(0.0, index=values.index)

    return (values - minimum) / (maximum - minimum)


def load_road_network(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Road network not found: {path}"
        )

    G = nx.read_graphml(path)

    print("Road network loaded successfully")
    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())

    return G


def calculate_edge_centrality(G):
    print("\nCalculating edge betweenness centrality...")

    # Convert the road network into an undirected graph.
    # Parallel road edges are represented by one node pair.

    simple_graph = nx.Graph()

    simple_graph.add_nodes_from(G.nodes(data=True))

    for u, v, data in G.edges(data=True):
        if u == v:
            continue

        length = float(data.get("length", 1.0))

        if simple_graph.has_edge(u, v):
            simple_graph[u][v]["length"] = min(
                simple_graph[u][v]["length"],
                length
            )
        else:
            simple_graph.add_edge(
                u, v, length=length
            )

    k = min(SAMPLE_SIZE, simple_graph.number_of_nodes())

    centrality = nx.edge_betweenness_centrality(
        simple_graph,
        k=k,
        normalized=True,
        weight="length",
        seed=42
    )

    print("Centrality calculation completed.")

    return {
    frozenset(edge): value
    for edge, value in centrality.items()
}


def extract_road_data(G, centrality):
    rows = []

    flood_fields = [
        "flood_exposure",
        "flood_risk",
        "flood_risk_score"
    ]

    for u, v, key, data in G.edges(
        keys=True, data=True
    ):
        if u == v:
            continue

        centrality_value = centrality.get(
            frozenset((u, v)), 0.0
        )

        # Use existing flood attributes only.
        flood_value = None

        for field in flood_fields:
            if field in data:
                try:
                    value = float(data[field])

                    if pd.notna(value):
                        flood_value = value
                        break
                except (ValueError, TypeError):
                    continue

        rows.append({
            "u": u,
            "v": v,
            "key": key,
            "road_name": str(
                data.get("name", "Unnamed Road")
            ),
            "road_type": str(
                data.get("highway", "Unknown")
            ),
            "length_m": float(
                data.get("length", 1.0)
            ),
            "edge_betweenness": centrality_value,
            "flood_risk": flood_value
        })

    return pd.DataFrame(rows)


def rank_critical_roads(df):
    df["centrality_score"] = normalize(
        df["edge_betweenness"]
    )

    has_flood_data = df["flood_risk"].notna().any()

    if has_flood_data:
        df["flood_risk_score"] = normalize(
            df["flood_risk"].fillna(0)
        )

        df["criticality_score"] = (
            CENTRALITY_WEIGHT * df["centrality_score"]
            + FLOOD_WEIGHT * df["flood_risk_score"]
        )

        df["risk_data_status"] = "Existing flood attributes"

    else:
        df["flood_risk_score"] = None

        # Without flood data, rank by actual network centrality.
        df["criticality_score"] = df["centrality_score"]

        df["risk_data_status"] = "Flood data unavailable"

    df = df.sort_values(
        by="criticality_score",
        ascending=False
    ).reset_index(drop=True)

    df["rank"] = range(1, len(df) + 1)

    return df


def main():
    print("\n======================================")
    print("       CRITICAL ROAD RANKING")
    print("======================================")

    G = load_road_network(GRAPH_PATH)

    centrality = calculate_edge_centrality(G)

    df = extract_road_data(G, centrality)

    df = rank_critical_roads(df)

    os.makedirs("outputs", exist_ok=True)

    df.to_csv(OUTPUT_PATH, index=False)

    print("\nRanking completed successfully!")

    print("\nTop 10 Critical Road Segments:")

    print(
        df[
            [
                "rank",
                "u",
                "v",
                "road_name",
                "edge_betweenness",
                "flood_risk",
                "criticality_score"
            ]
        ].head(10).to_string(index=False)
    )

    print("\nTotal road segments ranked:", len(df))

    if df["flood_risk"].notna().any():
        print("Flood risk: Existing attributes used.")
    else:
        print("Flood risk: Not available in the graph.")
        print("Ranking currently uses network centrality only.")

    print("\nCSV saved at:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
