"""
Ship Movement Model
===================
Desription here
"""

import numpy as np

from mesa import Model, DataCollector
from ship import Ship
from mesa.space import ContinuousSpace


class ShipModel(Model):
    """Ship model class. Handles agent creation, placement and scheduling."""

    def __init__(
        self,
        population=100,
        width=100,
        height=100,
        speed_range=(0.5, 1.5),
        vision=10,
        separation=2,
        cohere=0,
        separate=0.015,
        match=0,
        seed=None,
        ports=None
    ):
        """Create a new Ship model.

        Args:
            population: Number of ships in the simulation (default: 100)
            width: Width of the space (default: 100)
            height: Height of the space (default: 100)
            speed_range: Tuple specifying the min and max speed for ships (default: (0.5, 1.5))
            vision: How far each ship can see (default: 10)
            separation: Minimum distance between ships (default: 2)
            cohere: Weight of cohesion behavior (default: 0.03)
            separate: Weight of separation behavior (default: 0.015)
            match: Weight of alignment behavior (default: 0.05)
            seed: Random seed for reproducibility (default: None)
            ports: List of port coordinates [(x1, y1), (x2, y2), ...] (default: None)
        """
        super().__init__(seed=seed)

        # Model Parameters
        self.population = population
        self.vision = vision
        self.speed_range = speed_range
        self.separation = separation

        # Set up the space
        self.space = ContinuousSpace(width, height, torus=False)

        # Store flocking weights
        self.factors = {"cohere": cohere, "separate": separate, "match": match}

        # Define ports
        self.ports = ports if ports is not None else self.generate_random_ports()

        # Create and place the Ship agents
        self.make_agents()

        # For tracking statistics
        self.datacollector = DataCollector(
            agent_reporters={"pos": "pos"}
        )      
        self.datacollector.collect(self)  

    def generate_random_ports(self, num_ports=2):
        """Generate a list of random port coordinates."""
        return [
            (
                self.random.random() * self.space.x_max,
                self.random.random() * self.space.y_max
            )
            for _ in range(num_ports)
        ]

    def make_agents(self):
        """Create and place all ship agents randomly in a port."""
        for _ in range(self.population):
            # Select a random starting port
            start_port = self.random.choice(self.ports)
            pos = np.array(start_port)

            # Select a random destination port different from the starting port
            destination_port = self.random.choice([port for port in self.ports if port != start_port])

            # Assign a random speed within the speed range
            speed = self.random.uniform(*self.speed_range)

            # Create and place the agent
            agent = Ship(
                model=self,
                speed=speed,
                destination=np.array(destination_port),
                vision=self.vision,
                separation=self.separation,
                **self.factors,
            )
            self.space.place_agent(agent, pos)

    def step(self):
        """Run one step of the model.

        All agents are activated in random order using the AgentSet shuffle_do method.
        """
        self.agents.shuffle_do("step")
        self.datacollector.collect(self)