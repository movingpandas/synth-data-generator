import math
from heapq import heappush, heappop
import numpy as np
from shapely import vectorized


def create_occupancy_grid(map_width, map_height, resolution, obstacles, threshold=0):
    grid_width = int(map_width / resolution)
    grid_height = int(map_height / resolution)
    grid = np.zeros((grid_width, grid_height), dtype=int)

    xs = (np.arange(grid_width) + 0.5) * resolution
    ys = (np.arange(grid_height) + 0.5) * resolution
    xx, yy = np.meshgrid(xs, ys, indexing='ij')

    for obs in obstacles:
        if obs["type"] == "circle":
            shape = obs["circle"]
        else:
            shape = obs["polygon"]
        poly = shape.buffer(threshold)
        mask = vectorized.contains(poly, xx, yy)
        grid[mask] = 1

    return grid


def heuristic(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return path[::-1]


def astar(grid, start, goal):
    print(f'Grid shape = {grid.shape}')
    width, height = grid.shape
    open_set = []
    heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        current_f, current = heappop(open_set)
        if current == goal:
            return reconstruct_path(came_from, current)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                neighbor = (current[0] + dx, current[1] + dy)
                if 0 <= neighbor[0] < width and 0 <= neighbor[1] < height:
                    if grid[neighbor[0], neighbor[1]] == 1:
                        continue
                    tentative_g = g_score[current] + math.hypot(dx, dy)
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                        heappush(open_set, (f_score[neighbor], neighbor))
    return None


