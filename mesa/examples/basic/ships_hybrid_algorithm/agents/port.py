from mesa import Agent

class Port(Agent):
    def __init__(self, model):
        super().__init__(self, model)

    def step(self):
        """Ports remain static."""
        pass  # Ports remain static