"""
Ship Movement Model
===================
Desription here
"""

import numpy as np

from mesa import Model, DataCollector
from mesa.space import ContinuousSpace

from ship import Ship
from obstacle import Obstacle


class ShipModel(Model):
    """Ship model class. Handles agent creation, placement and scheduling."""

    def __init__(
        self,
        population=100,
        width=100,
        height=100,
        speed_range=(0.5, 1.5),
        vision=7,
        avoidance=0.5,
        cohere=0,
        separate=0.015,
        match=0,
        seed=None,
        ports=None,
        obstacles=None
    ):
        """Create a new Ship model.

        Args:
            population: Number of ships in the simulation 
            width: Width of the space 
            height: Height of the space 
            speed_range: Tuple specifying the min and max speed for ships 
            vision: How far each ship can see 
            avoidance: factor for avoiding obstacles
            cohere: Weight of cohesion behavior 
            separate: Weight of separation behavior 
            match: Weight of alignment behavior 
            seed: Random seed for reproducibility 
            ports: List of port coordinates [(x1, y1), (x2, y2), ...] 
        """
        super().__init__(seed=seed)

        # Model Parameters
        self.population = population
        self.vision = vision
        self.speed_range = speed_range
        self.avoidance = avoidance

        # Set up the space
        self.space = ContinuousSpace(width, height, torus=False)

        # Store flocking weights
        self.factors = {"cohere": cohere, "separate": separate, "match": match}

        # Define ports
        self.ports = ports if ports is not None else self.generate_random_ports()

        # Create and place the Ship agents
        self.make_agents()

        # Create and place obstacles
        self.create_obstacles(obstacles)

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

            # Assign a random max speed within the speed range
            max_speed = self.random.uniform(*self.speed_range)

            # Create and place the agent
            agent = Ship(
                model=self,
                max_speed=max_speed,
                destination=np.array(destination_port),
                vision=self.vision,
                avoidance=self.avoidance,
                **self.factors,
            )
            self.space.place_agent(agent, pos)

    def create_obstacles(self, obstacles):
        """Place random obstacles in the space."""
        for pos in obstacles:
            obstacle = Obstacle(self)
            pos = np.array(pos)
            self.space.place_agent(obstacle, pos)

    def step(self):
        """Run one step of the model.

        All agents are activated in random order using the AgentSet shuffle_do method.
        """
        self.agents.shuffle_do("step")
        self.datacollector.collect(self)