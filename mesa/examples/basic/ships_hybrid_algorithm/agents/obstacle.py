from mesa import Agent

class Obstacle(Agent):
    def __init__(self, model, shape):
        super().__init__(self, model)
        self.shape = shape  # Store the obstacle as a polygon

    def step(self):
        """Obstacles remain static."""
        pass