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
        obj_copy = Polygon(world=world, position=self.position, vertices=self.vertices, shape=self.shape,
                           color=self.color, rod=self.rod)
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
        b2PolygonShape.draw = State.__draw_polygon
        self.screen = None
        self.rod = None
        self.rodfix = None

    @staticmethod
    def __draw_polygon(obj, body, color, screen):
        vertices = [body.transform * v * PPM for v in obj.vertices]
        vertices = [(v[0], SCREEN_HEIGHT - v[1]) for v in vertices]
        pygame.draw.polygon(screen, color, vertices)  # inside rectangle
        pygame.draw.polygon(screen, BLACK, vertices, 5)  # boundary of rectangle

    def copy(self):
        state_copy = State()
        state_copy.objects = [obj.copy(state_copy.world) for obj in self.objects]
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

    def greedy(self, num_steps, prune_method, metric):
        actions = []
        for _ in range(num_steps):
            best_performance, best_push = self.greedy_step(prune_method, metric)
            self.push(best_push)
            assert self.count_soft_threshold() == best_performance
            actions.append(best_push)
        return best_performance, actions

    def greedy_step(self, prune_method, metric, sample_size=None):
        pushes = prune_method(self)
        if sample_size is None:
            indices = range(len(pushes))
        else:
            indices = np.random.choice(len(pushes), sample_size, replace=False)
        summary = self.save()
        best_push = None
        best_performance = metric()
        for i in indices:
            self.load(summary)
            action = pushes[i]
            self.push(action)
            curr_score = metric()
            if curr_score > best_performance:
                best_push = action
                best_performance = curr_score
        self.load(summary)
        return best_performance, best_push

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

    def push(self, action, path=None, display=False, check_reachable=True):
        start_pt, end_pt = action
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
                obj.fixture.shape.draw(obj.body, obj.color, self.screen)
            self.rodfix.shape.draw(self.rod, (0, 0, 0, 255), self.screen)
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
                    obj.fixture.shape.draw(obj.body, obj.color, self.screen)
                self.rodfix.shape.draw(self.rod, (0, 0, 0, 255), self.screen)
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

    def sample(self, num_steps, prune_method, metric, sampled, display=False, path=None):
        """Collect a sample of length num_steps > 1. For num_steps = 1, use greedy_step with a sample size."""
        assert num_steps > 1
        before_sampling = self.save()
        first_time = True
        unique = False
        while first_time or not unique:
            if not first_time:
                print("retry")
            actions = []
            for i in range(num_steps):
                pushes = prune_method(self)
                vec = random.choice(pushes)
                actions.extend(vec[0].tolist() + vec[1].tolist())
                if path is not None:
                    self.push(vec, display=display, path=path+str(i))
                else:
                    self.push(vec, display=display, path=path)
            final_score_sample = metric()
            self.load(before_sampling)
            first_time = False
            actions = tuple(actions)
            if actions not in sampled:
                unique = True
        return final_score_sample, actions

    def sample_best(self, num_sample, sample_func):
        best_result = 0
        best_push = None
        sampled = set()
        for _ in range(num_sample):
            sample_env = self.copy()
            result, action = sample_func(sample_env, sampled)
            print("action", action)
            assert action not in sampled
            sampled.add(action)
            if result > best_result:
                best_result = result
                best_push = action
        return best_result, best_push

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
            obj.fixture.shape.draw(obj.body, obj.color, self.screen)
        pygame.image.save(self.screen, path)
        pygame.display.quit()
        pygame.quit()


if __name__ == "__main__":
    num_samples = 5
    env = State()
    env.create_random_env(num_objs=5)
    print("starting score", env.count_soft_threshold())
    env.visualize("state.png")
    best_score, best_action = env.sample_best(num_sample=num_samples, sample_func=lambda e, sampled: e.sample(
        num_steps=2, prune_method=no_prune, metric=e.count_soft_threshold, sampled=sampled))
    print("best_score", best_score)
    print("best_action", best_action)

