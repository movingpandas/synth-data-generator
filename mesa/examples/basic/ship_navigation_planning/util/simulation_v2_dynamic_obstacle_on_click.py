import math
import random
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
from util.compute_path import create_occupancy_grid, astar
from util.dwa import motion, dwa_control_v2
from CFG import Config
from shapely.strtree import STRtree


class Simulation:
    def __init__(self):
        self.map_width = Config.MAP_WIDTH
        self.map_height = Config.MAP_HEIGHT
        self.resolution = Config.RESOLUTION
        self.threshold = Config.OBSTACLE_THRESHOLD

        self.start_point = Config.START_POINT
        self.goal_point = Config.GOAL_POINT

        self.lookahead = Config.LOOKAHEAD
        self.config = Config.MODEL_CONFIG

        self.start_idx = (
            int(self.start_point[0] / self.resolution),
            int(self.start_point[1] / self.resolution)
        )
        self.goal_idx = (
            int(self.goal_point[0] / self.resolution),
            int(self.goal_point[1] / self.resolution)
        )

        self.obstacles = []
        self.need_recompute_path = False
        self.global_path = None
        self.global_path_line = None
        self.buffered_obstacles = []
        self.obstacle_tree = None

        self.cid = None
        self.ax = self.fig = self.start_button = self.start_button_ax = self.anim = None

    def run(self):
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_xlim(0, self.map_width)
        self.ax.set_ylim(0, self.map_height)
        self.ax.set_title("Click to add obstacles. Then press Start.")

        self.ax.plot(self.start_point[0], self.start_point[1], "bo", markersize=10, label="Start")
        self.ax.plot(self.goal_point[0], self.goal_point[1], "go", markersize=10, label="Goal")

        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click_generate_circle)

        self.start_button_ax = plt.axes((0.81, 0.01, 0.1, 0.05))
        self.start_button = Button(self.start_button_ax, 'Start')
        self.start_button.on_clicked(self.start_callback)

        plt.show()

    @staticmethod
    def get_local_goal(state, global_path, current_wp_idx, lookahead=3.0):
        x, y = state[0], state[1]
        for idx in range(current_wp_idx, len(global_path)):
            wp = global_path[idx]
            if math.hypot(wp[0] - x, wp[1] - y) > lookahead:
                return wp, idx
        return global_path[-1], len(global_path) - 1

    @staticmethod
    def generate_random_polygon_at(x, y, avg_radius=10):
        num_vertices = random.randint(3, 10)
        angles = sorted(np.random.uniform(0, 2 * math.pi, num_vertices))
        vertices = []
        for angle in angles:
            radius = avg_radius * random.uniform(0.5, 1.5)
            vertices.append((x + radius * math.cos(angle), y + radius * math.sin(angle)))
        poly = Polygon(vertices)
        return {"type": "polygon", "polygon": poly}

    @staticmethod
    def generate_random_circle_at(x, y, radius=10):
        circle = Point(x, y).buffer(radius)
        return {"type": "circle", "circle": circle}

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        x, y = event.xdata, event.ydata
        obs = self.generate_random_polygon_at(x, y)
        self.obstacles.append(obs)

        x_obs, y_obs = obs["polygon"].exterior.xy
        self.ax.fill(x_obs, y_obs, color='red', alpha=0.5, edgecolor='black')
        plt.draw()

        self.need_recompute_path = True

    def on_click_generate_circle(self, event):
        if event.inaxes != self.ax:
            return
        x, y = event.xdata, event.ydata
        obs = self.generate_random_circle_at(x, y)
        self.obstacles.append(obs)

        x_obs, y_obs = obs["circle"].exterior.xy
        self.ax.fill(x_obs, y_obs, color='red', alpha=0.5, edgecolor='black')
        plt.draw()

        self.need_recompute_path = True

    def start_callback(self, event):
        self.start_button.ax.set_visible(False)
        plt.draw()
        self.run_simulation()

    def run_simulation(self):
        self.ax.clear()

        for obs in self.obstacles:
            if obs["type"] == "circle":
                shape = obs["circle"]
            else:
                shape = obs["polygon"]
            x_obs, y_obs = shape.exterior.xy
            self.ax.fill(x_obs, y_obs, color='red', alpha=0.5, edgecolor='black')

        self.ax.set_xlim(0, self.map_width)
        self.ax.set_ylim(0, self.map_height)
        self.ax.set_title("Hybrid A* + DWA Navigation (Dynamic Obstacles Enabled)")

        print('Computing A* global path...')

        grid = create_occupancy_grid(self.map_width, self.map_height, self.resolution,
                                     self.obstacles, self.threshold)
        global_path_indices = astar(grid, self.start_idx, self.goal_idx)
        print(global_path_indices)
        if global_path_indices is None:
            print("No global path found!")
            return

        self.global_path = [
            ((i + 0.5) * self.resolution, (j + 0.5) * self.resolution)
            for (i, j) in global_path_indices
        ]

        self.buffered_obstacles = [
            (obs["circle"] if obs["type"] == "circle" else obs["polygon"]).buffer(self.config["robot_radius"])
            for obs in self.obstacles
        ]
        self.obstacle_tree = STRtree(self.buffered_obstacles)

        gp_x, gp_y = zip(*self.global_path)
        self.global_path_line, = self.ax.plot(gp_x, gp_y, "c--", linewidth=2, label="Global Path (A*)")
        self.ax.plot(self.start_point[0], self.start_point[1], "bo", markersize=10, label="Start")
        self.ax.plot(self.goal_point[0], self.goal_point[1], "go", markersize=10, label="Goal")
        self.ax.legend()

        current_wp_idx = 0
        state = (self.start_point[0], self.start_point[1], 0.0, 0.0, 0.0)  # (x, y, theta, v, w)
        traj_x, traj_y = [state[0]], [state[1]]
        sim_time = 0.0

        traj_line, = self.ax.plot(traj_x, traj_y, "b-", linewidth=2, label="Trajectory")
        robot_marker, = self.ax.plot([state[0]], [state[1]], "ko", markersize=6, label="Robot")
        predicted_line, = self.ax.plot([], [], "m--", linewidth=1, label="Predicted Trajectory")
        info_text = self.ax.text(
            0.02, 0.98, '', transform=self.ax.transAxes, verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.5)
        )

        def update(frame):
            nonlocal state, traj_x, traj_y, sim_time, current_wp_idx

            if self.need_recompute_path:
                print("Recomputing global path due to dynamic obstacles...")
                grid = create_occupancy_grid(self.map_width, self.map_height, self.resolution,
                                             self.obstacles, self.threshold)
                new_start_idx = (
                    int(state[0] / self.resolution),
                    int(state[1] / self.resolution)
                )
                new_path_indices = astar(grid, new_start_idx, self.goal_idx)
                if new_path_indices is not None:
                    self.global_path = [
                        ((i + 0.5) * self.resolution, (j + 0.5) * self.resolution)
                        for (i, j) in new_path_indices
                    ]
                    current_wp_idx = 0
                    gp_x, gp_y = zip(*self.global_path)
                    self.global_path_line.set_data(gp_x, gp_y)
                    print("Global path updated.")
                else:
                    print("No global path found with the new obstacles!")

                self.buffered_obstacles = [
                    (obs["circle"] if obs["type"] == "circle" else obs["polygon"]).buffer(self.config["robot_radius"])
                    for obs in self.obstacles
                ]
                self.obstacle_tree = STRtree(self.buffered_obstacles)
                self.need_recompute_path = False

            if self.global_path is None or len(self.global_path) == 0:
                print("No valid global path. Stopping simulation.")
                self.anim.event_source.stop()
                return traj_line, robot_marker, predicted_line, info_text

            local_goal, current_wp_idx = self.get_local_goal(
                state, self.global_path, current_wp_idx, lookahead=self.lookahead
            )

            control, predicted_trajectory, cost_info = dwa_control_v2(
                state, self.config, self.obstacle_tree, self.buffered_obstacles, local_goal
            )
            state = motion(state, control[0], control[1], self.config["dt"])
            traj_x.append(state[0])
            traj_y.append(state[1])
            sim_time += self.config["dt"]

            traj_line.set_data(traj_x, traj_y)
            robot_marker.set_data([state[0]], [state[1]])
            pred_x = [s[0] for s in predicted_trajectory]
            pred_y = [s[1] for s in predicted_trajectory]
            predicted_line.set_data(pred_x, pred_y)

            info_str = (
                f"Time: {sim_time: .1f} s\n"
                f"Speed: {state[3]: .2f} m/s\n"
                f"Control: v = {control[0]: .2f} m/s, w = {np.rad2deg(control[1]): .1f} deg/s\n"
                f"Total Cost: {cost_info['total_cost']: .2f}\n"
                f"  → To-Goal: {cost_info['to_goal_cost']: .2f}\n"
                f"  → Speed: {cost_info['speed_cost']: .2f}\n"
                f"  → Obstacle: {cost_info['obstacle_cost']: .2f}\n"
                f"  → Turn: {cost_info['turn_cost']: .2f}"
            )
            info_text.set_text(info_str)

            if math.hypot(state[0] - self.goal_point[0], state[1] - self.goal_point[1]) < self.config["robot_radius"]:
                print("Final goal reached!")
                self.anim.event_source.stop()
            return traj_line, robot_marker, predicted_line, info_text

        self.anim = FuncAnimation(self.fig, update, frames=np.arange(0, 1000), interval=10, blit=False)
        plt.draw()

