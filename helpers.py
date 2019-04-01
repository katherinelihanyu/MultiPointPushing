import math
import numpy as np

def adjacent_location(location, dist=1.8, lower_bound=1, upper_bound=11):
    adj_pos = np.array([np.random.uniform(-dist, dist), np.random.uniform(-dist, dist)]) + np.array(location)
    adj_pos = np.clip(adj_pos, lower_bound, upper_bound)
    return adj_pos

def polygon(min_rad=math.sqrt(2) * 2 / 3, max_rad=math.sqrt(2), num_ver=6):
    """Return the vertices of a randomly generated polygon."""
    angles = sorted([np.random.uniform(0, 2 * math.pi) for _ in range(num_ver)])
    rad = [np.random.uniform(min_rad, max_rad) for _ in range(num_ver)]
    vertices = [[math.cos(angles[i]) * rad[i], math.sin(angles[i]) * rad[i]] for i in range(num_ver)]
    return vertices

def sigmoid(x):
  return 1 / (1 + math.exp(-x))
