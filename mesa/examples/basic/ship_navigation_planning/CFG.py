import numpy as np


class Config:
    MAP_WIDTH = 150
    MAP_HEIGHT = 150
    RESOLUTION = 1.0
    OBSTACLE_THRESHOLD = 3

    START_POINT = (10, 10)
    GOAL_POINT = (140, 140)

    LOOKAHEAD = 3

    MODEL_CONFIG = {
        "max_speed": 3.0,
        "min_speed": 0.1,
        "max_yaw_rate": np.deg2rad(15.0),
        "max_acceleration": 0.5,
        "max_delta_yaw_rate": np.deg2rad(15.0),
        "dt": 0.1,
        "predict_time": 1,
        "robot_radius": 1,
        "to_goal_cost_gain": 1,
        "speed_cost_gain": 1,
        "obstacle_cost_gain": 1.0,
        "turn_cost_gain": 1.0,
    }
