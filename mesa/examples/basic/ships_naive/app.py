from model import ShipModel
from obstacle import Obstacle 
from mesa.visualization import Slider, SolaraViz, make_space_component


def boid_draw(agent):
    if isinstance(agent, Obstacle):
        return {"color": "red", "size": 20}

    speed = agent.current_speed

    if speed < 0.1:
        return {"color": "blue", "size": 20}
    else: 
        return {"color": "green", "size": 20}


model_params = {
    "population": Slider(
        label="Number of of agents",
        value=20,
        min=1,
        max=100,
        step=1,
    ),
    "vision": Slider(
        label="Vision of agents (radius)",
        value=20,
        min=1,
        max=50,
        step=1,
    ),
}

model = ShipModel()

page = SolaraViz(
    model,
    components=[
        make_space_component(agent_portrayal=boid_draw, backend="matplotlib")],
    model_params=model_params,
    name="Ship Model",
)
page  # noqa