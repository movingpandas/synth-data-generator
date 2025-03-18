import logging
from datetime import datetime
import json

from model import ShipModel
from visualization import plot_simulation

# Configure logging
logging.basicConfig(
    filename="logs.log",
    filemode="w",  # Overwrites on each run; use "a" to append
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def run_simulation(config_file="config/config.json"):
    with open(config_file) as f:
        config = json.load(f)

    steps = config["simulation_steps"]

    model = ShipModel(
        width=config["width"],
        height=config["height"],
        num_ships=config["num_ships"],
        speed_range=config["speed_range"],
        ports=config["ports"],
        obstacles=config["obstacles"],
        dwa_config=config["dwa_config"],
        resolution=config["resolution"],
        obstacle_threshold=config["obstacle_threshold"],
        lookahead=config["lookahead"]
    )

    # Print agent counts by type
    for agent_type, agents in model.agents_by_type.items():
        logging.info(f'{agent_type}: {len(agents)}')

    # Run the simulation
    logging.info(f"{datetime.now()} Starting ...")
    for t in range(steps):
        logging.info(f"Step {t}")
        model.step()
    logging.info(f"{datetime.now()} Finished.")

    agent_df = model.datacollector.get_agent_vars_dataframe().dropna()
    df = agent_df.reset_index()

    # Save to CSV
    df.to_csv("ship_simulation_output.csv", index=False)
    logging.info("Saved simulation data to ship_simulation_output.csv")

    # Run visualization
    plot_simulation(df, config)


if __name__ == "__main__":
    run_simulation()