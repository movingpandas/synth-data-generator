"""A ship agent

Description here
"""

import numpy as np
import warnings

from mesa import Agent


class Ship(Agent):
    """A ship agent that moves towards a destination port."""

    def __init__(
        self,
        model,
        speed,
        destination,
        vision,
        separation,
        cohere=0.03,
        separate=0.015,
        match=0.05,
    ):
        """Create a new Ship agent.

        Args:
            model: Model instance the agent belongs to
            speed: Distance to move per step
            destination: numpy array of the destination port's coordinates
            vision: Radius to look around for nearby ships
            separation: Minimum distance to maintain from other ships
            cohere: Relative importance of matching neighbors' positions (default: 0.03)
            separate: Relative importance of avoiding close neighbors (default: 0.015)
            match: Relative importance of matching neighbors' directions (default: 0.05)
        """
        super().__init__(model)
        self.speed = speed
        self.destination = destination
        self.vision = vision
        self.separation = separation
        self.cohere_factor = cohere
        self.separate_factor = separate
        self.match_factor = match

    def step(self):
        """Move the ship towards its destination."""
        # Compute the direction vector towards the destination
        direction_to_destination = self.destination - self.pos
        distance_to_destination = np.linalg.norm(direction_to_destination)

        # Check if the ship has reached its destination
        if distance_to_destination < self.speed:
            self.model.space.move_agent(self, self.destination)
            return

        # Normalize the direction vector and scale by speed
        movement_vector = (direction_to_destination / distance_to_destination) * self.speed

        # Add small random variations to the movement direction
        random_variation = np.random.normal(0, 0.1, 2)  # Gaussian noise with mean 0 and std deviation 0.1
        movement_vector += random_variation

        # Normalize the movement vector again after adding noise
        movement_vector /= np.linalg.norm(movement_vector)
        movement_vector *= self.speed

        # Calculate new position
        new_pos = self.pos + movement_vector

        # Move the ship
        self.model.space.move_agent(self, new_pos)