import matplotlib.pyplot as plt
import pandas as pd
import json
from shapely.geometry import Polygon

def plot_simulation(df, config, output_image="simulation_plot.png"):
    ports = config["ports"]
    obstacles = config["obstacles"]

    # Create figure
    plt.figure(figsize=(10, 10))

    # Plot ports
    for x, y in ports:
        plt.scatter(x, y, color='blue', marker='s', label="Port" if 'Port' not in plt.gca().get_legend_handles_labels()[1] else "")

    # Plot obstacles (islands)
    for island in obstacles:
        island_shape = Polygon(island)
        x, y = island_shape.exterior.xy
        plt.fill(x, y, color="brown", alpha=0.6, label="Island" if 'Island' not in plt.gca().get_legend_handles_labels()[1] else "")

    # Track whether we've added the A* legend entry already
    a_star_legend_added = False

    # Plot ship trajectories
    for agent_id in df["AgentID"].unique():
        ship_data = df[df["AgentID"] == agent_id]

        # Extract A* path from the first entry of each ship
        a_star_path = ship_data.iloc[0]["AStarPath"]

        # Set label only for the first A* path
        a_star_label = "A* Path" if not a_star_legend_added else None
        a_star_legend_added = True  # Ensure only the first A* path gets labeled

        # Plot A* path as a dashed line
        a_star_x, a_star_y = zip(*a_star_path)
        plt.plot(a_star_x, a_star_y, linestyle="dashed", color="gray", alpha=0.7, label=a_star_label)

        # Plot trajectory as a solid line
        plt.plot(ship_data["x"], ship_data["y"], linestyle="-", alpha=0.7, label=f"Ship {agent_id}")

    # Labels and settings
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("DWA Ship Movement Trajectories and A* Paths")

    # Move legend outside the plot
    plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    
    plt.grid(True)

    # Adjust layout to make space for the legend
    plt.tight_layout()

    # Save plot as an image
    plt.savefig(output_image, bbox_inches="tight")  # Ensure the legend is included
    print(f"Saved simulation plot to {output_image}")
