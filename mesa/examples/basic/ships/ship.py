import numpy as np
from math import pi
from mesa import Agent
from obstacle import Obstacle
from dynamic_window_approach import dwa_control, motion, Config

class Ship(Agent):
    """A ship agent that moves towards a destination port using the Dynamic Window Approach."""

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
        self.current_speed = 0.1  # Start with zero velocity
        self.acceleration = 1  # Acceleration per step
        self.deceleration = 0.1  # Deceleration per step
        self.destination = destination
        self.vision = vision
        #self.avoidance = avoidance
        #self.cohere_factor = cohere
        #self.separate_factor = separate
        #self.match_factor = match
        
        # Ship's state [x, y, yaw, velocity, yaw_rate]
        self.state = np.array([0.0, 0.0, pi / 8.0, self.current_speed, 0.0])
        
        # DWA configuration using ship's speed and acceleration settings
        self.dwa_config = Config()
        self.dwa_config.max_speed = self.max_speed
        self.dwa_config.max_accel = self.acceleration
        self.dwa_config.robot_radius = 1
        #self.dwa_config.dt = 0.5

    def step(self):
        """Move the ship towards its destination using DWA."""
        
        
        direction_to_destination = self.destination - self.pos
        distance_to_destination = np.linalg.norm(direction_to_destination)
        
        if distance_to_destination == 0: 
            #print('Stopped ...')
            return
        
    
        #print('Step ...')
        if distance_to_destination < (self.current_speed*self.dwa_config.dt):
            #print('Moving to destination ...')
            self.move_to_destination()
            return
        
        # Get obstacles in vision range
        obstacles_in_range = self.find_obstacles_in_view_range()
        if obstacles_in_range:
            obstacle_positions = np.array([obs.pos for obs in obstacles_in_range])
        else:
            obstacle_positions = np.zeros((1, 2))  # Avoid empty array error
        
        # Apply DWA to compute best velocity and yaw rate
        self.state[0] = float(self.pos[0])
        self.state[1] = float(self.pos[1])
        x, _ = dwa_control(self.state, self.dwa_config, self.destination, obstacle_positions)
        
        # Update the ship's pos based on DWA motion model
        self.state = motion(self.state, x, self.dwa_config.dt)
        self.pos = (float(self.state[0]),float(self.state[1]))
        self.current_speed = float(self.state[3])
        
        # Move the ship in the simulation space
        self.move_position()
    
    def move_position(self):
        """Update the ship's position in the simulation space."""
        new_pos = np.array([self.pos[0], self.pos[1]])
        self.model.space.move_agent(self, new_pos)
    
    def move_to_destination(self):    
        """Move the ship directly to its destination."""
        self.pos[0], self.pos[1] = self.destination
        self.model.space.move_agent(self, self.destination)        
    
    def find_obstacles_in_view_range(self):
        """Find obstacles within the ship's vision range."""
        obstacles_in_range = [
            neighbor for neighbor in self.model.space.get_neighbors(
                self.pos, self.vision, include_center=True)
            if isinstance(neighbor, Obstacle)  # Check if the neighbor is an obstacle
        ]
        
        return obstacles_in_range
