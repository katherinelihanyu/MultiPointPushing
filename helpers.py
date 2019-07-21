import math
import numpy as np

def adjacent_location(location, dist=1.8, lower_bound=1, upper_bound=11):
    adj_pos = np.array([np.random.uniform(-dist, dist), np.random.uniform(-dist, dist)]) + np.array(location)
    adj_pos = np.clip(adj_pos, lower_bound, upper_bound)
    return adj_pos

def no_prune(env):
	pt_lst = []
	for ind in range(len(env.objects)*256):
		obj_ind = ind // 256
		theta1 = ((ind % 256) // 16) * 3.14 * 2 / 16
		theta2 = (ind % 16) * 3.14 * 2 / 16
		if theta1 == theta2:
			continue
		pt1 = (math.cos(theta1) * env.objects[obj_ind].bounding_circle_radius, math.sin(theta1) * env.objects[obj_ind].bounding_circle_radius)
		pt2 = (math.cos(theta2) * env.objects[obj_ind].bounding_circle_radius, math.sin(theta2) * env.objects[obj_ind].bounding_circle_radius)
		pts = parametrize_by_bounding_circle(np.array(pt1) + np.array([env.objects[obj_ind].body.position[0], env.objects[obj_ind].body.position[1]]),
											 np.array(pt2) - np.array(pt1),
											 np.array([env.objects[obj_ind].body.position[0], env.objects[obj_ind].body.position[1]]),
											 env.objects[obj_ind].bounding_circle_radius+0.1)
		if pts is not None:
			pt_lst.append(pts)
	return pt_lst

def normalize(vector):
	"""
	helper function: 
	input: vector (x, y) force vector
	output: vector (x, y) force vector with normalized magnitude 1
	"""
	mag = math.sqrt(vector[0] ** 2 + vector[1] ** 2)+1e-6
	return vector[0] / mag, vector[1] / mag

def parametrize_by_bounding_circle(start_pt, vector, centroid, bounding_circle_radius):
	"""parametrize as p1 to p2"""
	point = (start_pt[0] - centroid[0], start_pt[1] - centroid[1])
	a = (vector[0]**2 + vector[1]**2) + 1e-6
	b = (2 * point[0] * vector[0] + 2 * point[1] * vector[1])
	c = (point[0] ** 2 + point[1] ** 2 - bounding_circle_radius ** 2)
	if (b**2 - 4 * a * c) < 0:
		print("unable to parametrize by bounding circle: line of force does not touch bounding circle")
		return None
	else:
		t1 = (-b + math.sqrt(b**2 - 4 * a * c))/(2*a)
		t2 = (-b - math.sqrt(b**2 - 4 * a * c))/(2*a)
		p1 = (point[0] + t2 * vector[0], point[1] + t2 * vector[1])
		p2 = (point[0] + t1 * vector[0], point[1] + t1 * vector[1])
		return [np.array(normalize([p1[0], p1[1]])) * bounding_circle_radius + np.array(centroid), np.array(normalize([p2[0], p2[1]])) * bounding_circle_radius + np.array(centroid)]

def polygon(min_rad=math.sqrt(2) * 2 / 3, max_rad=math.sqrt(2), num_ver=6):
    """Return the vertices of a randomly generated polygon."""
    angles = sorted([np.random.uniform(0, 2 * math.pi) for _ in range(num_ver)])
    rad = [np.random.uniform(min_rad, max_rad) for _ in range(num_ver)]
    vertices = [[math.cos(angles[i]) * rad[i], math.sin(angles[i]) * rad[i]] for i in range(num_ver)]
    return vertices

def rotatePt(point, vector):
	radius = unitVector2Degree(vector)/180*math.pi
	x = point[0]*math.cos(radius)-point[1]*math.sin(radius)
	y = point[0]*math.sin(radius)+point[1]*math.cos(radius)
	return (x, y)

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def unitVector2Degree(vector):
	vector = normalize(vector)
	return math.atan2(vector[1], vector[0])*180/math.pi
