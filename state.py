import imageio
import os
import pygame
import random
import matplotlib.pyplot as plt

from Box2D import (b2Distance, b2PolygonShape, b2Transform, b2World)
from helpers import *

"""Display variables."""
SCREEN_WIDTH, SCREEN_HEIGHT = 720, 720
PPM = 60.0  # pixels per meter
WHITE = (255, 255, 255, 255)
BLACK = (0, 0, 0, 0)
num_classes = 8
color_list = plt.cm.get_cmap('Accent')(np.linspace(0, 1, num_classes))
COLORS = [(int(255*c[0]), int(255*c[1]), int(255*c[2]), int(255*c[3])) for c in color_list]

"""Physics variables."""
TIME_STEP = 0.1
WORLD = b2World(gravity=(0, 0), doSleep=True)
FRICTION = 0.5
# 'POLYGON_SHAPES' is a list of (minimum radius, maximum radius, number of vertices) tuples.
POLYGON_SHAPES = [(1, 1, 15), (0.75, 1, 12), (0.5, 1, 9), (0.25, 1, 6)]
DAMPING_FACTOR = 1 - ((1 - 0.5) / 3)


class Polygon:
    def __init__(self, position=None, vertices=None, shape=0, color=None, rod=False):
        if vertices is None:
            self.vertices = np.array(polygon(*POLYGON_SHAPES[shape]))
        else:
            self.vertices = np.array(vertices)
        if position is None:
            self.position = np.random.uniform(low=4, high=8, size=2)
        else:
            self.position = position
        if color is None:
            self.color = random.choice(COLORS)
        else:
            self.color = color
        if rod:
            self.body = WORLD.CreateKinematicBody(position=self.position, allowSleep=False)
        else:
            self.body = WORLD.CreateDynamicBody(position=self.position.tolist(), allowSleep=False)
        self.fixture = self.body.CreatePolygonFixture(density=1, vertices=self.vertices.tolist(), friction=FRICTION)
        self.bounding_circle_radius = math.sqrt(max(self.vertices[:, 0]**2 + self.vertices[:, 1]**2))
    
    def destroy(self):
        self.body.DestroyFixture(self.fixture)
        WORLD.DestroyBody(self.body)

    def distance(self, other_polygon):
        transform1 = b2Transform()
        transform1.Set(self.body.position, self.body.angle)
        transform2 = b2Transform()
        transform2.Set(other_polygon.body.position, other_polygon.body.angle)
        dist = b2Distance(
            shapeA=self.fixture.shape, shapeB=other_polygon.fixture.shape, transformA=transform1, transformB=transform2)[2]
        return dist

    def overlap(self, other_polygon):
        return self.distance(other_polygon) <= 0


class State:
    def __init__(self, objects=[]):
        self.objects = objects
        b2PolygonShape.draw = self.__draw_polygon

    def __draw_polygon(self, polygon, body, color):
        vertices = [(body.transform * v) * PPM for v in polygon.vertices]
        vertices = [(v[0], SCREEN_HEIGHT - v[1]) for v in vertices]
        pygame.draw.polygon(self.screen, color, vertices)  # inside rectangle
        pygame.draw.polygon(self.screen, BLACK, vertices, 5)  # boundary of rectangle

    def clear(self):
        """Remove all objects."""
        for obj in self.objects:
            obj.destroy()
        self.objects = []

    def count_soft_threshold(self):
        count = 0.0
        for i in range(len(self.objects)):
            min_dist = 1e2
            for j in range(i+1, len(self.objects)):
                dist = self.objects[i].distance(self.objects[j])
                if dist < min_dist:
                    min_dist = dist
            count += (sigmoid(min_dist*10)-0.5)*2  # assume threshold=0.3. Want f(0)=0, f(0.3)=1
        return count

    def create_random_env(self, num_objs=3, shape=0, max_iter_limit=10000):
        self.clear()
        while len(self.objects) < num_objs:
            self.clear()
            self.objects.append(Polygon(shape=shape, color=COLORS[0]))
            for i in range(1, num_objs):
                for _ in range(max_iter_limit):
                    original_pos = adjacent_location(self.objects[-1].body.position) # place object close to the last object
                    obj = Polygon(position=original_pos, shape=shape, color=COLORS[i % len(COLORS)])
                    overlapped = False
                    for other_obj in self.objects:
                        if other_obj.overlap(obj):
                            overlapped = True
                            break 
                    if overlapped:
                        obj.body.DestroyFixture(obj.fixture)
                        WORLD.DestroyBody(obj.body)
                    else:
                        self.objects.append(obj)
                        break
                if len(self.objects) <= i:
                    break

    def load(self, summary):
        """Load environment defined by summary (an output from save)"""
        for i, obj in enumerate(self.objects):
           obj.body.position[0], obj.body.position[1], obj.body.angle, obj.body.linearVelocity[0], obj.body.linearVelocity[1], obj.body.angularVelocity = summary[i]

    def push(self, start_pt, end_pt, path=None, display=False, check_reachable=True):
        if display:
            images = []
            if not os.path.exists(path):
                os.makedirs(path)
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.iconify()
            self.screen.fill(WHITE)
        start_pt = np.array(start_pt)
        end_pt = np.array(end_pt)
        self.rod = WORLD.CreateKinematicBody(position=(start_pt[0], start_pt[1]), allowSleep=False)
        self.rodfix = self.rod.CreatePolygonFixture(vertices=[(0.1, 0.1), (-0.1, 0.1), (-0.1, -0.1), (0.1, -0.1)])
        vector = np.array(normalize(end_pt - np.array(start_pt)))
        # reachability check
        if check_reachable:
            vertices_lst = [(0.0, 0.1), (0.0, -0.1), (-0.3, -0.1), (-0.3, 0.1)]
            test_rod = Polygon(position=(start_pt[0], start_pt[1]), vertices=[rotatePt(pt, vector) for pt in vertices_lst], rod=True)
            while (np.count_nonzero(np.array([obj.distance(test_rod) for obj in self.objects]) <= 0) > 0):
                start_pt -= 0.1 * vector
                test_rod.destroy()
                test_rod = Polygon(position=(start_pt[0], start_pt[1]), vertices=[rotatePt(pt, vector) for pt in vertices_lst], rod=True)
            test_rod.destroy()
        self.rod.linearVelocity[0] = vector[0]
        self.rod.linearVelocity[1] = vector[1]
        self.rod.angularVelocity = 0.0
        timestamp = 0
        # display
        if display:
            for obj in self.objects:
                obj.fixture.shape.draw(obj.fixture.shape, obj.body, obj.color)
            self.rodfix.shape.draw(self.rodfix.shape, self.rod, (0, 0, 0, 255))
            img_path = os.path.join(path, '0.png')
            pygame.image.save(self.screen, img_path)
            images.append(imageio.imread(img_path))
            os.remove(img_path)
        first_contact = -1
        while (timestamp < 100):
            if first_contact == -1:
                for i in range(len(self.objects)):
                    # if object is moving, classify it as contacted
                    if (self.objects[i].body.linearVelocity[0] ** 2 + self.objects[i].body.linearVelocity[1] ** 2 > 0.001):
                        first_contact = i
            for obj in self.objects:
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
            WORLD.Step(TIME_STEP, 10, 10)
            timestamp += 1
            # display
            if display:
                self.screen.fill((255, 255, 255, 255))
                for obj in self.objects:
                    obj.fixture.shape.draw(obj.fixture.shape, obj.body, obj.color)
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

    def sample(self, num_steps, prune_method, metric, display=False, path=None):
        """Collect a sample of length num_steps"""
        actions = np.zeros((num_steps,4))
        for i in range(num_steps):
            vec = random.choice(prune_method(self))
            actions[i] = np.hstack((vec[0],vec[1])).flatten()
            self.push(vec[0], vec[1], display=display, path=os.path.join(path, str(i)))
        actions_tuple = tuple(actions.flatten())
        final_score = metric()
        print("after", self.count_soft_threshold())
        assert final_score == self.count_soft_threshold()
        return final_score, actions_tuple

    def save(self):
        """Save information about current state in a dictionary in sum_path/env.json"""
        summary = np.array([[obj.body.position[0], obj.body.position[1], obj.body.angle, obj.body.linearVelocity[0], obj.body.linearVelocity[1], obj.body.angularVelocity] for obj in self.objects])
        return summary

    def visualize(self, path):
        """Capture an image of the current state.
        path: .../something.png
        """
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.iconify()
        self.screen.fill(WHITE)
        for obj in self.objects:
            obj.fixture.shape.draw(obj.fixture.shape, obj.body, obj.color)
        pygame.image.save(self.screen, path)
        pygame.display.quit()
        pygame.quit()


if __name__ == "__main__":
    env = State()
    env.create_random_env(num_objs=5)
    print("before count", env.count_soft_threshold())
    final_score, actions_tuple = env.sample(num_steps=3, prune_method=no_prune, metric=env.count_soft_threshold, display=True, path="./push")
    print("after count", final_score)
    print(actions_tuple)
    

