
import os
import pandas as pd
import matplotlib.pyplot as plt

# Create output folder if it does not exist
os.makedirs("outputs", exist_ok=True)

# -----------------------------------
# 1. Top 10 Critical Roads
# -----------------------------------

critical_file = "outputs/top_10_critical_roads.csv"

# Use the existing ranking file if top_10 file is unavailable
if not os.path.exists(critical_file):
    critical_file = "outputs/road_edge_betweenness_ranking.csv"

if os.path.exists(critical_file):

    critical = pd.read_csv(critical_file)

    # Check whether the centrality column exists
    if "centrality" not in critical.columns:
        print("ERROR: 'centrality' column not found.")
        print("Available columns:", critical.columns.tolist())

    else:
        critical = critical.sort_values(
            by="centrality",
            ascending=False
        ).head(10)

        plt.figure(figsize=(10, 5))

        plt.bar(
            range(1, len(critical) + 1),
            critical["centrality"]
        )

        plt.xlabel("Road segment rank")
        plt.ylabel("Edge betweenness centrality")
        plt.title("Top 10 Most Critical Roads")

        plt.xticks(range(1, len(critical) + 1))
        plt.tight_layout()

        plt.savefig(
            "outputs/top_10_critical_roads.png"
        )

        plt.show()

else:
    print("Critical roads CSV file not found.")


# -----------------------------------
# 2. Evacuation Distance Comparison
# -----------------------------------

connectivity_file = "outputs/connectivity_results.csv"

if os.path.exists(connectivity_file):

    data = pd.read_csv(connectivity_file)

    required_columns = [
        "normal_distance_m",
        "flood_distance_m"
    ]

    if all(col in data.columns for col in required_columns):

        plot_data = data.dropna(
            subset=required_columns
        )

        if not plot_data.empty:

            plt.figure(figsize=(10, 5))

            plt.plot(
                range(1, len(plot_data) + 1),
                plot_data["normal_distance_m"],
                marker="o",
                label="Normal network"
            )

            plt.plot(
                range(1, len(plot_data) + 1),
                plot_data["flood_distance_m"],
                marker="o",
                label="Flood scenario"
            )

            plt.xlabel("Origin-destination pair")
            plt.ylabel("Shortest route distance (metres)")
            plt.title(
                "Evacuation Route Distance: Before vs After Flood"
            )

            plt.legend()
            plt.tight_layout()

            plt.savefig(
                "outputs/evacuation_comparison.png"
            )

            plt.show()

        else:
            print("No valid distance records available.")

    else:
        print("Required distance columns are missing.")
        print("Available columns:", data.columns.tolist())

else:
    print("Connectivity results file not found.")
    print("Run evacuation_connectivity.py first.")


print("\nVisualization process completed.")
