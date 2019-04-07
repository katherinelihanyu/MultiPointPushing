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
FRICTION = 0.5
# 'POLYGON_SHAPES' is a list of (minimum radius, maximum radius, number of vertices) tuples.
POLYGON_SHAPES = [(1, 1, 15), (0.75, 1, 12), (0.5, 1, 9), (0.25, 1, 6)]
DAMPING_FACTOR = 1 - ((1 - 0.5) / 3)


class Polygon:
    def __init__(self, world, position=None, vertices=None, shape=0, color=None, rod=False):
        self.world = world
        self.shape=shape
        self.rod = rod
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
            self.body = self.world.CreateKinematicBody(position=self.position, allowSleep=False)
        else:
            self.body = self.world.CreateDynamicBody(position=self.position.tolist(), allowSleep=False)
        self.fixture = self.body.CreatePolygonFixture(density=1, vertices=self.vertices.tolist(), friction=FRICTION)
        self.bounding_circle_radius = math.sqrt(max(self.vertices[:, 0]**2 + self.vertices[:, 1]**2))

    def copy(self, world):
        obj_copy = Polygon(world=world,position=self.position, vertices=self.vertices,shape=self.shape,color=self.color,
                           rod=self.rod)
        return obj_copy

    def destroy(self):
        self.body.DestroyFixture(self.fixture)
        self.world.DestroyBody(self.body)

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
        self.world = b2World(gravity=(0, 0), doSleep=True)
        self.objects = objects
        b2PolygonShape.draw = self.__draw_polygon
        self.screen = None
        self.rod = None
        self.rodfix = None

    def __draw_polygon(self, obj, body, color, screen):
        vertices = [body.transform * v * PPM for v in obj.vertices]
        vertices = [(v[0], SCREEN_HEIGHT - v[1]) for v in vertices]
        pygame.draw.polygon(screen, color, vertices)  # inside rectangle
        pygame.draw.polygon(screen, BLACK, vertices, 5)  # boundary of rectangle
    
    def __draw_line(self, start, end, color=BLACK, width=5):
        start*=PPM
        start = (start[0], SCREEN_HEIGHT-start[1])
        end*=PPM
        end = (end[0], SCREEN_HEIGHT-end[1])
        pygame.draw.line(self.screen, color, start, end, width)

    def copy(self):
        state_copy = State()
        state_copy.objects =[obj.copy(state_copy.world) for obj in self.objects]
        return state_copy

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
            self.objects.append(Polygon(world=self.world, shape=shape, color=COLORS[0]))
            for i in range(1, num_objs):
                for _ in range(max_iter_limit):
                    original_pos = adjacent_location(self.objects[-1].body.position) # place object close to the last object
                    obj = Polygon(world=self.world, position=original_pos, shape=shape, color=COLORS[i % len(COLORS)])
                    overlapped = False
                    for other_obj in self.objects:
                        if other_obj.overlap(obj):
                            overlapped = True
                            break 
                    if overlapped:
                        obj.body.DestroyFixture(obj.fixture)
                        self.world.DestroyBody(obj.body)
                    else:
                        self.objects.append(obj)
                        break
                if len(self.objects) <= i:
                    break
        for i, obj in enumerate(self.objects):
            assert obj.body.angle == 0.0
            assert obj.body.linearVelocity[0] == 0.0
            assert obj.body.linearVelocity[1] == 0.0
            assert obj.body.angularVelocity == 0.0

    def load(self, summary):
        """Load environment defined by summary (an output from save)"""
        if self.rod:
            self.rod.DestroyFixture(self.rodfix)
            self.world.DestroyBody(self.rod)
            self.rod = None

        for i, obj in enumerate(self.objects):
            obj.body.position[0], obj.body.position[1], obj.body.angle = summary[i]
            obj.body.linearVelocity[0] = 0.0
            obj.body.linearVelocity[1] = 0.0
            obj.body.angularVelocity = 0.0

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
        self.rod = self.world.CreateKinematicBody(position=(start_pt[0], start_pt[1]), allowSleep=False)
        self.rodfix = self.rod.CreatePolygonFixture(vertices=[(0.1, 0.1), (-0.1, 0.1), (-0.1, -0.1), (0.1, -0.1)])
        vector = np.array(normalize(end_pt - np.array(start_pt)))
        # reachability check
        if check_reachable:
            vertices_lst = [(0.0, 0.1), (0.0, -0.1), (-0.3, -0.1), (-0.3, 0.1)]
            test_rod = Polygon(world=self.world, position=(start_pt[0], start_pt[1]), vertices=[rotatePt(pt, vector) for pt in vertices_lst], rod=True)
            while np.count_nonzero(np.array([obj.distance(test_rod) for obj in self.objects]) <= 0) > 0:
                start_pt -= 0.1 * vector
                test_rod.destroy()
                test_rod = Polygon(world=self.world, position=(start_pt[0], start_pt[1]), vertices=[rotatePt(pt, vector) for pt in vertices_lst], rod=True)
            test_rod.destroy()
        self.rod.linearVelocity[0] = vector[0]
        self.rod.linearVelocity[1] = vector[1]
        self.rod.angularVelocity = 0.0
        timestamp = 0
        # display
        if display:
            for obj in self.objects:
                obj.fixture.shape.draw(obj, obj.body, obj.color, self.screen)
            self.rodfix.shape.draw(self.rodfix.shape, self.rod, (0, 0, 0, 255), self.screen)
            # self.__draw_line(start_pt, end_pt)
            img_path = os.path.join(path, '0.png')
            pygame.image.save(self.screen, img_path)
            images.append(imageio.imread(img_path))
            # os.remove(img_path)
        first_contact = -1
        while timestamp < 100:
            if first_contact == -1:
                for i in range(len(self.objects)):
                    # if object is moving, classify it as contacted
                    if self.objects[i].body.linearVelocity[0] ** 2 + self.objects[i].body.linearVelocity[1] ** 2 > 0.001:
                        first_contact = i
            for obj in self.objects:
                obj.body.linearVelocity[0] *= DAMPING_FACTOR
                obj.body.linearVelocity[1] *= DAMPING_FACTOR
                obj.body.angularVelocity *= DAMPING_FACTOR
            if math.sqrt(np.sum((start_pt - np.array(self.rod.position)) ** 2)) < 4:
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
                for obj in self.objects:
                    obj.fixture.shape.draw(obj.fixture.shape, obj.body, obj.color, self.screen)
                self.rodfix.shape.draw(self.rodfix.shape, self.rod, (0, 0, 0, 255), self.screen)
                # self.__draw_line(start_pt, end_pt)
                img_path = os.path.join(path, str(timestamp) + '.png')
                pygame.image.save(self.screen, img_path)
                images.append(imageio.imread(img_path))
                # os.remove(img_path)
        # display
        if display:
            pygame.display.quit()
            pygame.quit()
            imageio.mimsave(os.path.join(path, 'push.gif'), images)
        return first_contact

    def sample(self, num_steps, prune_method, metric, display=False, path=None):
        """Collect a sample of length num_steps"""
        before_sampling = self.save()
        # print("before_sampling", before_sampling)
        # print("count", self.count_soft_threshold())
        actions = []
        for i in range(num_steps):
            vec = random.choice(prune_method(self))
            actions.extend(vec[0].tolist() + vec[1].tolist())
            if path != None:
                self.push(vec[0], vec[1], display=display, path=path+str(i))
            else:
                self.push(vec[0], vec[1], display=display, path=path)
            # print("after_sampling", self.save())
            # print("count", self.count_soft_threshold())
        final_score = metric()
        final_state = self.save()
        self.load(before_sampling)
        return final_score, tuple(actions), final_state

    def save(self):
        """Save information about current state in a dictionary in sum_path/env.json"""
        summary = np.array([[obj.body.position[0], obj.body.position[1], obj.body.angle] for obj in self.objects])
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

    def print_vertices(self):
        pass


if __name__ == "__main__":
    env = State()
    env.create_random_env(num_objs=5)
    # final_score, actions_tuple = env.sample(num_steps=1, prune_method=no_prune, metric=env.count_soft_threshold, display=True, path="./draw")
    env.visualize("./no_angle.png")
    print("BEFORE")
    print(env.save())
    print("BEOFRE VERTICES")
    print(env.objects[0].vertices)
    env.objects[0].body.angle = 1.03780246e+00
    env.visualize(("./angle.png"))
    print("AFTER VERTICES")
    print(env.save())
    print(env.objects[0].vertices)
