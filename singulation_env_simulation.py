import logging
import math
import time
import copy

import pygame
from pygame.locals import (QUIT, KEYDOWN, K_ESCAPE)
import numpy as np
import pickle
import imageio

import Box2D  
from Box2D.b2 import (world, polygonShape, circleShape, staticBody, dynamicBody, kinematicBody)

import matplotlib.pyplot as plt
import os
from helper import *
import json
import random
from policies import *
from prune import *
from keras.models import load_model
from mpc import *

PPM = 60.0  # pixels per meter
TIME_STEP = 0.1
SCREEN_WIDTH, SCREEN_HEIGHT = 720, 720
GROUPS = [(1, 1, 15), (0.75, 1, 12), (0.5, 1, 9), (0.25, 1, 6)]
num_classes = 8
base = plt.cm.get_cmap('Accent')
color_list = base(np.linspace(0, 1, num_classes))
COLORS = [(int(255*c[0]), int(255*c[1]), int(255*c[2]), int(255*c[3])) for c in color_list]
WHITE = (255, 255, 255, 255)
FRICTION = 0.5
DAMPING_FACTOR = 1 - ((1 - 0.5) / 3)


class Action:
    def __init__(self, vector, point):
        """
        action that consists of:
        vector: (x, y) force vector
        point: (x, y) point of contact
        all relative to the local origin of polygon.
        """
        self.vector = normalize(vector)
        self.point = point

    def __eq__(self, other):
        """
        check if vector == vector, point == point
        """
        return self.vector == other.vector and self.point == self.point


class Polygon:
    def __init__(self, body, fixtures, vertices, color=WHITE):
        """body: polygon shape (dynamicBody)
        fixture: fixture
        vertices: list of relative coordinates
        """
        self.body = body
        self.fixtures = fixtures
        self.vertices = vertices
        self.color = color
        self.original_pos = np.array(self.body.position)
        # self.bounding_circle_radius = math.sqrt(max(self.vertices[:, 0]**2 + self.vertices[:, 1]**2))

    def test_overlap(self, other_polygon):
        if self.dist(other_polygon) > 0:
            return False
        return True

    def dist(self, other_polygon):
        """do not work on a polygon and a rod"""
        shape1 = self.fixtures[0].shape
        shape2 = other_polygon.fixtures[0].shape
        transform1 = Box2D.b2Transform()
        pos1 = self.body.position
        angle1 = self.body.angle
        transform1.Set(pos1, angle1)
        transform2 = Box2D.b2Transform()
        pos2 = other_polygon.body.position
        angle2 = other_polygon.body.angle
        transform2.Set(pos2, angle2)
        point_a, point_b, distance, iterations = Box2D.b2Distance(
            shapeA=shape1, shapeB=shape2, transformA=transform1, transformB=transform2)
        return distance

    def dist_rod(self, rod_fix, rod_body):
        shape1 = self.fixtures[0].shape
        shape2 = rod_fix.shape
        transform1 = Box2D.b2Transform()
        pos1 = self.body.position
        angle1 = self.body.angle
        transform1.Set(pos1, angle1)
        transform2 = Box2D.b2Transform()
        pos2 = rod_body.position
        angle2 = rod_body.angle
        transform2.Set(pos2, angle2)
        point_a, point_b, distance, iterations = Box2D.b2Distance(
            shapeA=shape1, shapeB=shape2, transformA=transform1, transformB=transform2)
        return distance


class SingulationEnv:
    def __init__(self):
        self.world = world(gravity=(0, 0), doSleep=True)
        self.objs = []
        self.rod = None
        self.bounding_convex_hull = np.array([])
        self.centroid = (0, 0)
        self.bounding_circle_radius = 0
        self.screen = None
        polygonShape.draw = self.my_draw_polygon
        circleShape.draw = self.my_draw_circle

    def my_draw_polygon(self, polygon, body, color):
        vertices = [(body.transform * v) * PPM for v in polygon.vertices]
        vertices = [(v[0], SCREEN_HEIGHT - v[1]) for v in vertices]
        pygame.draw.polygon(self.screen, color, vertices)  # inside rectangle
        pygame.draw.polygon(self.screen, (0, 0, 0, 0), vertices, 5)  # boundary of rectangle

    def my_draw_circle(self, circle, body, color):
        position = body.transform * circle.pos * PPM
        position = (position[0], SCREEN_HEIGHT - position[1])
        pygame.draw.circle(self.screen, color, [int(
            x) for x in position], int(circle.radius * PPM))

    def create_random_env(self, num_objs=3, group=0):
        assert num_objs >= 1
        # clear environment
        if len(self.objs) > 0:
            for obj in self.objs:
                for fix in obj.fixtures:
                    obj.body.DestroyFixture(fix)
                self.world.DestroyBody(obj.body)
        self.objs = []
        for i in range(num_objs):
            # create shape
            vertices = generatePolygon(GROUPS[group][0], GROUPS[group][1], GROUPS[group][2])
            raw_com = np.array(compute_centroid(vertices))
            print("raw_com", raw_com)
            vertices = (vertices - raw_com)
            bounding_r = math.sqrt(max(vertices[:, 0]**2 + vertices[:, 1]**2))
            print("bounding_r", bounding_r)
            vertices = vertices / bounding_r
            # place object
            if len(self.objs) == 0:
                original_pos = np.array([np.random.uniform(4, 8), np.random.uniform(4, 8)])
                body = self.world.CreateDynamicBody(position=original_pos.tolist(), allowSleep=False)
                fixture = body.CreatePolygonFixture(density=1, vertices=vertices.tolist(), friction=FRICTION)
                self.objs.append(Polygon(body, [fixture], vertices, COLORS[i]))
            else:
                max_iter = 10000
                while True:
                    if max_iter <= 0:
                        raise RuntimeError("max iter reaches")
                    max_iter -= 1
                    # place object close to the last object
                    original_pos = np.array([np.random.uniform(-1.8, 1.8), np.random.uniform(-1.8, 1.8)]) + np.array(self.objs[-1].body.position)
                    original_pos = np.clip(original_pos, 1, 11)
                    body = self.world.CreateDynamicBody(position=original_pos.tolist(), allowSleep=False)
                    fixture = body.CreatePolygonFixture(density=1, vertices=vertices.tolist(), friction=FRICTION)
                    curr_polygon = Polygon(body, [fixture], vertices, COLORS[i % len(COLORS)])
                    overlap = False
                    for obj in self.objs:
                        if obj.test_overlap(curr_polygon):
                            overlap = True
                            break
                    if overlap:
                        body.DestroyFixture(fixture)
                        self.world.DestroyBody(body)
                    else:
                        self.objs.append(curr_polygon)
                        break
        self.bounding_convex_hull = create_convex_hull(np.concatenate([obj.vertices+obj.original_pos for obj in self.objs]))
        self.centroid = compute_centroid(self.bounding_convex_hull.tolist())
        self.bounding_circle_radius = math.sqrt(max(
            (self.bounding_convex_hull - np.array(self.centroid))[:, 0]**2
            + (self.bounding_convex_hull - np.array(self.centroid))[:, 1]**2))

    def create_random_env_wrapper(self, num_objs=3, group=0):
        while True:
            try:
                self.create_random_env(num_objs,group)
                return
            except RuntimeError:
                pass

    def visualize(self, path):
        """Capture an image of the current state.
        path: .../something.png
        """
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.iconify()
        self.screen.fill(WHITE)
        for obj in self.objs:
            for fix in obj.fixtures:
                fix.shape.draw(fix.shape, obj.body, obj.color)
        pygame.image.save(self.screen, path)
        pygame.display.quit()
        pygame.quit()

    def step(self, start_pt, end_pt, path, display=False, check_reachable=True):
        if display:
            images = []
            if not os.path.exists(path):
                os.makedirs(path)
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.iconify()
            self.screen.fill(WHITE)
        start_pt = np.array(start_pt)
        end_pt = np.array(end_pt)
        self.rod = self.world.CreateKinematicBody(position=(start_pt[0], start_pt[1]), allowSleep=False)
        self.rodfix = self.rod.CreatePolygonFixture(vertices=[(0.1, 0.1), (-0.1, 0.1), (-0.1, -0.1), (0.1, -0.1)])
        vector = np.array(normalize(end_pt - np.array(start_pt)))
        # reachability check
        if check_reachable:
            vertices_lst = [(0.0, 0.1), (0.0, -0.1), (-0.3, -0.1), (-0.3, 0.1)]
            testrod = self.world.CreateKinematicBody(position=(start_pt[0], start_pt[1]), allowSleep=False)
            testrodfix = testrod.CreatePolygonFixture(vertices=[rotatePt(pt, vector) for pt in vertices_lst])
            while (np.count_nonzero(np.array([o.dist_rod(testrodfix, testrod) for o in self.objs]) <= 0) > 0):
                start_pt -= 0.1 * vector
                testrod.DestroyFixture(testrodfix)
                self.world.DestroyBody(testrod)
                testrod = self.world.CreateKinematicBody(position=(start_pt[0], start_pt[1]), allowSleep=False)
                testrodfix = testrod.CreatePolygonFixture(vertices=[rotatePt(pt, vector) for pt in vertices_lst])
            testrod.DestroyFixture(testrodfix)
            self.world.DestroyBody(testrod)
        self.rod.linearVelocity[0] = vector[0]
        self.rod.linearVelocity[1] = vector[1]
        self.rod.angularVelocity = 0.0
        timestamp = 0
        # display
        if display:
            for obj in self.objs:
                for fix in obj.fixtures:
                    fix.shape.draw(fix.shape, obj.body, obj.color)
            self.rodfix.shape.draw(self.rodfix.shape, self.rod, (0, 0, 0, 255))
            img_path = os.path.join(path, '0.png')
            pygame.image.save(self.screen, img_path)
            images.append(imageio.imread(img_path))
            os.remove(img_path)
        first_contact = -1
        while (timestamp < 100):
            if first_contact == -1:
                for i in range(len(self.objs)):
                    # if object is moving, classify it as contacted
                    if (self.objs[i].body.linearVelocity[0] ** 2 + self.objs[i].body.linearVelocity[1] ** 2 > 0.001):
                        first_contact = i
            for obj in self.objs:
                obj.body.linearVelocity[0] *= DAMPING_FACTOR
                obj.body.linearVelocity[1] *= DAMPING_FACTOR
                obj.body.angularVelocity *= DAMPING_FACTOR
            if (math.sqrt(np.sum((start_pt - np.array(self.rod.position)) ** 2)) < 4):
                vector = normalize((end_pt + 1e-8) - (start_pt + 1e-8))
                self.rod.linearVelocity[0] = vector[0]
                self.rod.linearVelocity[1] = vector[1]
            else:
                self.rod.linearVelocity[0] = 0
                self.rod.linearVelocity[1] = 0
            self.world.Step(TIME_STEP, 10, 10)
            timestamp += 1
            # display
            if display:
                self.screen.fill((255, 255, 255, 255))

                for obj in self.objs:
                    for fix in obj.fixtures:
                        fix.shape.draw(fix.shape, obj.body, obj.color)

                self.rodfix.shape.draw(self.rodfix.shape, self.rod, (0, 0, 0, 255))
                img_path = os.path.join(path, str(timestamp) + '.png')
                pygame.image.save(self.screen, img_path)
                images.append(imageio.imread(img_path))
                os.remove(img_path)
        # display
        if display:
            pygame.display.quit()
            pygame.quit()
            imageio.mimsave(os.path.join(path, 'push.gif'), images)
        return first_contact

    def step_area(self, start_pt, end_pt, gripper_width, path, display=False, check_reachable=True):
        start_pt = np.array(start_pt)
        end_pt = np.array(end_pt)
        vertices_lst=[(0.1, gripper_width/2), (-0.1, gripper_width/2), (-0.1, -gripper_width/2), (0.1, -gripper_width/2)]
        vector = np.array(normalize(end_pt - start_pt))
        self.rod = self.world.CreateKinematicBody(position=(start_pt[0], start_pt[1]), allowSleep=False)
        self.rodfix = self.rod.CreatePolygonFixture(vertices=[rotatePt(pt, vector) for pt in vertices_lst])
        # reachability check
        if check_reachable:
            while (np.count_nonzero(np.array([o.dist_rod(self.rodfix, self.rod) for o in self.objs]) <= 0) > 0):
                start_pt -= 0.1 * vector
                self.rod.DestroyFixture(self.rodfix)
                self.world.DestroyBody(self.rod)
                self.rod = self.world.CreateKinematicBody(position=(start_pt[0], start_pt[1]), allowSleep=False)
                self.rodfix = self.rod.CreatePolygonFixture(vertices=[rotatePt(pt, vector) for pt in vertices_lst])
        self.rod.linearVelocity[0] = vector[0]
        self.rod.linearVelocity[1] = vector[1]
        self.rod.angularVelocity = 0.0
        timestamp = 0
        first_contact = -1
        while (timestamp < 100):
            if first_contact == -1:
                for i in range(len(self.objs)):
                    if (self.objs[i].body.linearVelocity[0] ** 2 + self.objs[i].body.linearVelocity[1] ** 2 > 0.001):
                        first_contact = i
            for obj in self.objs:
                obj.body.linearVelocity[0] *= DAMPING_FACTOR
                obj.body.linearVelocity[1] *= DAMPING_FACTOR
                obj.body.angularVelocity *= DAMPING_FACTOR
            if (math.sqrt(np.sum((start_pt - np.array(self.rod.position))**2)) < 4):
                vector = normalize((end_pt+1e-8) - (start_pt+1e-8))
                self.rod.linearVelocity[0] = vector[0]
                self.rod.linearVelocity[1] = vector[1]
            else:
                self.rod.linearVelocity[0] = 0
                self.rod.linearVelocity[1] = 0
                break
            self.world.Step(TIME_STEP, 8, 3)
            timestamp += 1
        return first_contact

    def count_threshold(self, threshold=0.3):
        count = 0
        for i in range(len(self.objs)):
            isolated = True
            for j in range(len(self.objs)):
                if i != j:
                    shape1 = self.objs[i].fixtures[0].shape
                    shape2 = self.objs[j].fixtures[0].shape
                    transform1 = Box2D.b2Transform()
                    pos1 = self.objs[i].body.position
                    angle1 = self.objs[i].body.angle
                    transform1.Set(pos1, angle1)
                    transform2 = Box2D.b2Transform()
                    pos2 = self.objs[j].body.position
                    angle2 = self.objs[j].body.angle
                    transform2.Set(pos2, angle2)
                    pointA, pointB, distance, iterations = Box2D.b2Distance(shapeA=shape1, shapeB=shape2, transformA=transform1, transformB=transform2)
                    if distance < threshold:
                        isolated = False
            if isolated:
                count += 1
        return count

    def count_soft_threshold(self):
        count = 0.0
        for i in range(len(self.objs)):
            min_dist = 1e2
            for j in range(i+1, len(self.objs)):
                shape1 = self.objs[i].fixtures[0].shape
                shape2 = self.objs[j].fixtures[0].shape
                transform1 = Box2D.b2Transform()
                pos1 = self.objs[i].body.position
                angle1 = self.objs[i].body.angle
                transform1.Set(pos1, angle1)
                transform2 = Box2D.b2Transform()
                pos2 = self.objs[j].body.position
                angle2 = self.objs[j].body.angle
                transform2.Set(pos2, angle2)
                pointA, pointB, distance, iterations = Box2D.b2Distance(shapeA=shape1, shapeB=shape2, transformA=transform1, transformB=transform2)
                if distance < min_dist:
                    min_dist = distance
            # assume threshold = 0.3. Want f(0) = 0, f(0.3)=1
            count += (sigmoid(min_dist*10) - 0.5) * 2
        return count

    def collect_data_summary(self, start_pt, end_pt, golden_summary=None, first_step=False, img_path=None, display=False, sum_path=None):
        summary = {}
        abs_start_pt = np.array(start_pt)
        abs_end_pt = np.array(end_pt)
        summary["start pt"] = abs_start_pt.tolist()
        summary["end pt"] = abs_end_pt.tolist()
        for i in range(len(self.objs)):
            summary[str(i) + " dist to pushing line"] = pointToLineDistance(abs_start_pt, abs_end_pt, self.objs[i].body.position)
            summary[str(i) + " original pos"] = np.array(self.objs[i].body.position).tolist()
            summary[str(i) + " project dist"] = projectedPtToStartDistance(abs_start_pt, abs_end_pt, self.objs[i].body.position)
            summary[str(i) + " vertices"] = np.array(self.objs[i].vertices).tolist()
        summary["count soft threshold before push"] = self.count_soft_threshold()
        first_contact = self.step(start_pt, end_pt, path=img_path, display=display)
        for i in range(len(self.objs)):
            summary[str(i) + " change of pos"] = euclidean_dist(self.objs[i].body.position, self.objs[i].original_pos)
        summary["count soft threshold after push"] = self.count_soft_threshold()
        summary["first contact object"] = first_contact
        if sum_path is not None:
            with open(os.path.join(sum_path, 'summary.json'), 'w') as f:
                json.dump(summary, f)
        return summary

    def reset(self):
        if self.rod:
            self.rod.DestroyFixture(self.rodfix)
            self.world.DestroyBody(self.rod)
            self.rod = None
        for obj in self.objs:
            obj.body.position[0] = obj.original_pos[0]
            obj.body.position[1] = obj.original_pos[1]
            obj.body.angle = 0.0
            obj.body.linearVelocity[0] = 0.0
            obj.body.linearVelocity[1] = 0.0
            obj.body.angularVelocity = 0.0

    def save_env(self, sum_path=None):
        """Save information about current state in a dictionary in sum_path/env.json"""
        summary = {}
        summary["num_objects"] = len(self.objs)
        summary["count soft threshold"] = self.count_soft_threshold()
        for i in range(len(self.objs)):
            summary[str(i) + " original pos"] = np.array(self.objs[i].body.position).tolist()
            summary[str(i) + " angle"] = self.objs[i].body.angle
            summary[str(i) + " vertices"] = np.array(self.objs[i].vertices).tolist()
        if sum_path != None:
            with open(os.path.join(sum_path, 'env.json'), 'w') as f:
                json.dump(summary, f)
        return summary

    def load_env(self, dic):
        for i in range(dic["num_objects"]):
            original_pos = np.array(dic[str(i)+" original pos"])
            vertices = np.array(dic[str(i)+" vertices"])
            body = self.world.CreateDynamicBody(position=original_pos.tolist(), angle=dic.get(str(i)+" angle", 0.0), allowSleep=False)
            fixture = body.CreatePolygonFixture(density=1, vertices=vertices.tolist(), friction=FRICTION)
            self.objs.append(Polygon(body, [fixture], vertices, COLORS[i % len(COLORS)]))

    def save_curr_position(self):
        position = {}
        for i in range(len(self.objs)):
            position[i] = [self.objs[i].body.position[0], self.objs[i].body.position[1], self.objs[i].body.angle]
        return position

    def load_position(self, position):
        if self.rod:
            self.rod.DestroyFixture(self.rodfix)
            self.world.DestroyBody(self.rod)
            self.rod = None
        for i in range(len(self.objs)):
            self.objs[i].body.position[0] = position[i][0]
            self.objs[i].body.position[1] = position[i][1]
            self.objs[i].body.angle = position[i][2]

    def best_sequential_sample(self, num_samples, prune_method, reuse, max_step, data_path, reuse_path=None, metric="count soft threshold", open_loop=False):
        best_result = 0
        best_action = None
        best_sample = None
        best_info = None
# 		import pdb; pdb.set_trace()
        summary = self.save_env()
        first_step_pt_lst = prune_method(self)
        complete_pt_lst = list(first_step_pt_lst)
        for i in range(num_samples):
            sample_path = os.path.join(data_path, "sample"+str(i))
            if not os.path.exists(sample_path):
                os.makedirs(sample_path)
            before = self.save_env()
            result, action, info = collect_sequential_sample(complete_pt_lst, summary, self, reuse, max_step, sample_path, metric, open_loop)
            if before != self.save_env():
                print("before != self.save_env()")
            if result > best_result:
                best_result = result
                best_action = action
                best_sample = i
                best_info = info
        if reuse:
            with open(os.path.join(reuse_path, "best_sample.json"), 'w') as f:
                json.dump({"best_sample": best_sample}, f)
        else:
            with open(os.path.join(data_path,"best_sample.json"), 'w') as f:
                json.dump({"best_sample": best_sample}, f)
        return best_result, best_action, best_info

    def prune_best(self, prune_method, metric="count threshold", position=None):
        pt_lst = prune_method(self)
        best_pt = None
        best_sep = -1e2
        print("len(pt_lst)", len(pt_lst))
        info = {}
        self.visualize("./A.png")
        for pts in pt_lst[:1000]:
            if position is None:
                self.reset()
            else:
                self.load_position(position)
            summary = self.collect_data_summary(pts[0], pts[1], None)
            if summary[metric +" after push"] - summary[metric + " before push"] >= best_sep:
                best_pt = pts
                best_sep = summary[metric +" after push"] - summary[metric + " before push"]
                info[metric + " before push"] = summary[metric + " before push"]
                info[metric + " after push"] = summary[metric + " after push"]
                info["env after push"] = self.save_env()
                self.visualize("./B.png")
        if position is None:
            self.reset()
        else:
            self.load_position(position)
        return best_pt, info

    def select_random(self, prune_method):
        pt_lst = prune_method(self)
        return random.choice(pt_lst)

    # Forward dynamics model
    def generate_state_action_pairs(self,path=None,display=False):
        state_action = []
        for o in self.objs:
            state_action.append(o.body.position[0])
            state_action.append(o.body.position[1])
            state_action.append(o.body.angle)
            for v in o.vertices:
                state_action.append(v[0])
                state_action.append(v[1])
        best_pts = self.select_random(com_only)
        if best_pts is None:
            return [], []
        best_summary = self.collect_data_summary(best_pts[0], best_pts[1], img_path=path, display=display)
        vec = normalize(np.array(best_pts[1]) - np.array(best_pts[0]))
        state_action.append(best_pts[0][0])
        state_action.append(best_pts[0][1])
        state_action.append(vec[0])
        state_action.append(vec[1])
        next_state = []
        for o in self.objs:
            next_state.append(o.body.position[0])
            next_state.append(o.body.position[1])
            next_state.append(o.body.angle)
            for v in o.vertices:
                next_state.append(v[0])
                next_state.append(v[1])
        if best_summary["first contact object"] == -1:
            return [], []
        return state_action, next_state

    def reset_env_five_objects(self, v):
        if self.rod:
            self.rod.DestroyFixture(self.rodfix)
            self.world.DestroyBody(self.rod)
            self.rod = None
        for i in range(len(self.objs)):
            obj = self.objs[i]
            index = int(33*i)
            obj.body.position[0] = float(v[index])
            obj.body.position[1] = float(v[index+1])
            obj.body.angle = float(v[index+2])
            obj.body.linearVelocity[0] = 0.0
            obj.body.linearVelocity[1] = 0.0
            obj.body.angularVelocity = 0.0

# if __name__ == "__main__":
    # data_path = "/nfs/diskstation/katherineli/sampling"
    # create_initial_envs(50,10,data_path)
    # env = SingulationEnv()
    # env.create_random_env_wrapper()
    # env.visualize("./norm2.png")
    # pts = no_prune(env)[10]
    # env.step(pts[0],pts[1],".",display=True)
