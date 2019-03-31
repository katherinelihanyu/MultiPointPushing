import math
import numpy as np


def compute_centroid(vertices):
    """
    helper function:

    input:
    vertices: a list of vertices of a polygon
    under the assumption that all vertices are ordered either clockwise/counterclockwise

    output:
    centroid: position of (x, y) tuple of the polygon relative to the local origin of polygon.
    """
    c_x = 0
    c_y = 0
    area = 0
    n = len(vertices)
    for i in range(n):
        curr = vertices[(i - n) % n]
        next = vertices[(i + 1 - n) % n]
        diff = (curr[0] * next[1] - curr[1] * next[0])
        c_x += (curr[0] + next[0]) * diff
        c_y += (curr[1] + next[1]) * diff
        area += diff
    area = area / 2
    c_x = c_x / (6 * area)
    c_y = c_y / (6 * area)
    return c_x, c_y

def normalize_vertices(vertices):
    raw_com = np.array(compute_centroid(vertices))
    vertices = (vertices - raw_com)
    bounding_r = math.sqrt(max(vertices[:, 0] ** 2 + vertices[:, 1] ** 2))
    vertices = vertices / bounding_r
    return vertices

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
