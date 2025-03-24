from mesa import Agent

class Obstacle(Agent):
    """An obstacle that ships must avoid."""

    def __init__(self, model):
        """Create an obstacle agent."""
        super().__init__(model)
        #self.pos = pos  # Position of the obstacle