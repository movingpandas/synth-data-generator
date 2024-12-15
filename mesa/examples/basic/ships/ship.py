"""A ship agent

Description here
"""

import numpy as np
import warnings

from mesa import Agent

from obstacle import Obstacle



class Ship(Agent):
    """A ship agent that moves towards a destination port."""

    def __init__(
        self,
        model,
        max_speed,
        destination,
        vision,
        avoidance,
        cohere=0.03,
        separate=0.015,
        match=0.05,
    ):
        """Create a new Ship agent.

        Args:
            model: Model instance the agent belongs to
            max_speed: Maximum speed of the ship
            destination: numpy array of the destination port's coordinates
            vision: Radius to look around for nearby ships
            separation: factor for avoidance of obstacles
            cohere: Relative importance of matching neighbors' positions (default: 0.03)
            separate: Relative importance of avoiding close neighbors (default: 0.015)
            match: Relative importance of matching neighbors' directions (default: 0.05)
        """
        super().__init__(model)
        self.max_speed = max_speed
        self.current_speed = 0.0  # Start with zero velocity
        self.acceleration = 0.05  # Acceleration per step
        self.deceleration = 0.1  # Deceleration per step
        self.destination = destination
        self.vision = vision
        self.avoidance = avoidance
        self.cohere_factor = cohere
        self.separate_factor = separate
        self.match_factor = match

    def step(self):
        """Move the ship towards its destination and avoid obstacles."""
        direction_to_destination = self.destination - self.pos
        distance_to_destination = np.linalg.norm(direction_to_destination)

        if distance_to_destination < self.current_speed:
            self.move_to_destination()
            return
        
        if distance_to_destination < self.current_speed*10:
            self.current_speed = max(0, self.current_speed - self.deceleration)
        # Accelerate to max speed if not near the destination
        elif self.current_speed < self.max_speed:
            self.current_speed = min(self.max_speed, self.current_speed + self.acceleration)


        # Normalize the direction vector towards the destination
        movement_vector = direction_to_destination / distance_to_destination


        # Add random variation to the movement direction
        variation = np.random.normal(0, 0.05, size=2)  
        movement_vector += variation
        
        # Avoid obstacles if they are within vision range
        obstacles_in_range = self.find_obstacles_in_view_range()
        if obstacles_in_range:
            avoidance_vector = self.compute_avoidance_vector(obstacles_in_range, movement_vector)
            movement_vector += avoidance_vector

        movement_vector /= np.linalg.norm(movement_vector)  # Re-normalize after adding avoidance

        self.move_position(movement_vector)

    def move_position(self, movement_vector):
        new_pos = self.pos + movement_vector * self.current_speed
        self.model.space.move_agent(self, new_pos)

    def compute_avoidance_vector(self, obstacles_in_range, movement_vector):
        avoidance_vector = np.zeros(2)

        obstacles_in_range.sort(key=lambda obs: np.linalg.norm(self.pos - obs.pos))
        obstacle = obstacles_in_range[0]

        # Calculate the vector from the obstacle to the ship
        vector_to_obstacle = self.pos - obstacle.pos
        distance_to_obstacle = np.linalg.norm(vector_to_obstacle)

        if distance_to_obstacle > 0:
            # Avoidance strength increases as obstacles get closer
            avoidance_strength = min(1.0, self.avoidance / distance_to_obstacle)

            # Check alignment of obstacle with movement direction
            dot_product = np.dot(
                    vector_to_obstacle / distance_to_obstacle, 
                    movement_vector / np.linalg.norm(movement_vector)
                )
            if dot_product > 0.8:  # Strong alignment
                avoidance_strength += 0.2

            # Add scaled avoidance vector
            avoidance_vector += (vector_to_obstacle / distance_to_obstacle) * avoidance_strength
        return avoidance_vector

    def move_to_destination(self):    
        new_pos = self.destination
        self.model.space.move_agent(self, new_pos)        

    def find_obstacles_in_view_range(self):
        obstacles_in_range = [
            neighbor for neighbor in self.model.space.get_neighbors(
                self.pos, self.vision, include_center=True)
            if isinstance(neighbor, Obstacle)  # Check if the neighbor is an obstacle
        ]
        
        return obstacles_in_range