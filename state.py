import imageio
import os
import pygame
import random
import matplotlib.pyplot as plt

from Box2D import (b2Distance, b2PolygonShape, b2Transform, b2World)
from helpers import *
from math import isclose

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
    def __init__(self, world, position=None, vertices=None, shape=0, color=None, rod=False, info=None):
        # position: numpy array of size (2,)
        # vertices: numpy array of size (15, 2) for group 0
        # color: 4-tuple
        # info: numpy array of size (38,) for group 0
        self.world = world
        if info is None:
            self.shape = shape
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
        else:
            self.shape = info[0]
            self.rod = bool(info[1])
            self.position = info[2:4]
            self.color = info[4:8]
            self.vertices = np.reshape(info[8:], (-1, 2))
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

    def equal(self, obj):
        return np.array_equal(self.position, obj.position) and np.array_equal(self.vertices, obj.vertices) \
               and self.shape == obj.shape and self.rod == obj.rod and np.array_equal(self.color, obj.color)

    def overlap(self, other_polygon):
        return self.distance(other_polygon) <= 0

    def save(self):
        # return a size (38,) numpy array containing the following info in order:
        # obj.shape -> int
        # obj.rod -> boolean cast to int
        # position -> (2,)
        # color -> (4,)
        # vertices -> (15, 2) flatten
        info = np.array([self.shape, int(self.rod)] + self.position.tolist() + list(self.color))
        return np.hstack((info, self.vertices.flatten()))


class State:
    def __init__(self, objects=[], summary=None):
        self.world = b2World(gravity=(0, 0), doSleep=True)
        if summary is None:
            self.objects = objects
        else:
            self.objects = []
            num_objs = int(summary[-1])
            assert (summary.shape[0]-1) % num_objs == 0
            arr_len = summary.shape[0]//num_objs
            self.objects = [Polygon(self.world, info=summary[arr_len*i:arr_len*(i+1)]) for i in range(num_objs)]
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
        state_copy.objects = [object.copy(state_copy.world) for object in self.objects]
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

    def equal(self, state):
        return len(self.objects) == len(state.objects) \
               and all([self.objects[i].equal(state.objects[i]) for i in range(len(self.objects))])

    def greedy(self, num_steps, prune_method, metric):
        actions = []
        for _ in range(num_steps):
            best_performance, best_push, best_state = self.greedy_step(prune_method, metric)
            self.push(best_push)
            if not isclose(best_performance, self.count_soft_threshold(), abs_tol=1e-3):
                print("GREEDY NOT REPRODUCIBLE. Expected: %s; actual: %s" % (best_performance, self.count_soft_threshold()))
            actions.append(best_push)
        return best_performance, actions

    def greedy_step(self, prune_method, metric, sample_size=None):
        pushes = prune_method(self)
        if sample_size is None:
            indices = range(len(pushes))
        else:
            indices = np.random.choice(len(pushes), sample_size, replace=False)
        summary = self.save_positions()
        best_push = None
        best_performance = metric()
        best_state = None
        for i in indices:
            self.load_positions(summary)
            action = pushes[i]
            self.push(action)
            curr_score = metric()
            if curr_score > best_performance:
                best_push = action
                best_performance = curr_score
                best_state = self.save_positions()
        self.load_positions(summary)
        return best_performance, best_push, best_state

    def load_positions(self, summary):
        """Load environment defined by summary (an output from save)"""
        if summary.shape[0] > len(self.objects):
            if not self.rod:
                self.rod = self.world.CreateKinematicBody(position=(summary[len(self.objects)][0], summary[len(self.objects)][1]), allowSleep=False)
                self.rodfix = self.rod.CreatePolygonFixture(vertices=[(0.1, 0.1), (-0.1, 0.1), (-0.1, -0.1), (0.1, -0.1)])
            self.rod.position[0], self.rod.position[1], self.rod.angle = summary[len(self.objects)]
            self.rod.linearVelocity[0] = 0.0
            self.rod.linearVelocity[1] = 0.0
            self.rod.angularVelocity = 0.0
        elif self.rod:
            self.rod.DestroyFixture(self.rodfix)
            self.world.DestroyBody(self.rod)
            self.rod = None
        for i, obj in enumerate(self.objects):
            obj.body.position[0], obj.body.position[1], obj.body.angle = summary[i]
            obj.body.linearVelocity[0] = 0.0
            obj.body.linearVelocity[1] = 0.0
            obj.body.angularVelocity = 0.0

    def push(self, action, path=None, display=False, save_summary=False, save_frames=False, check_reachable=True):
        start_pt, end_pt = action
        if path is not None and not os.path.exists(path):
            os.makedirs(path)
        if display:
            images = []
            img_folder = os.path.join(path, "image")
            if not os.path.exists(img_folder):
                os.makedirs(img_folder)
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.iconify()
            self.screen.fill(WHITE)
        if save_summary:
            summary_folder = os.path.join(path, "summary")
            if not os.path.exists(summary_folder):
                os.makedirs(summary_folder)
            self.save(os.path.join(summary_folder, "i.npy"))
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
        if display:
            for obj in self.objects:
                obj.fixture.shape.draw(obj.body, obj.color, self.screen)
            self.rodfix.shape.draw(self.rod, (0, 0, 0, 255), self.screen)
            img_path = os.path.join(img_folder, '0.png')
            pygame.image.save(self.screen, img_path)
            images.append(imageio.imread(img_path))
            if not save_frames:
                os.remove(img_path)
        if save_summary:
            self.save_positions(os.path.join(summary_folder, "0.npy"))
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
            if display:
                self.screen.fill((255, 255, 255, 255))
                for obj in self.objects:
                    obj.fixture.shape.draw(obj.body, obj.color, self.screen)
                self.rodfix.shape.draw(self.rod, (0, 0, 0, 255), self.screen)
                img_path = os.path.join(img_folder, '%d.png'%timestamp)
                pygame.image.save(self.screen, img_path)
                images.append(imageio.imread(img_path))
                if not save_frames:
                    os.remove(img_path)
            if save_summary:
                self.save_positions(os.path.join(summary_folder, "%d.npy"%timestamp))
        if display:
            pygame.display.quit()
            pygame.quit()
            imageio.mimsave(os.path.join(img_folder, 'push.gif'), images)
        return first_contact

    def sample(self, num_steps, prune_method, metric, sampled, display=False, save_summary=False, path=None):
        """Collect a sample of length num_steps > 1. For num_steps = 1, use greedy_step with a sample size."""
        assert num_steps > 1
        before_sampling = self.save_positions()
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
                    self.push(vec, display=display, save_summary=save_summary, path=path+str(i))
                else:
                    self.push(vec, display=display, save_summary=save_summary, path=path)
                if i == 0:
                    first_step_end_state = self.save_positions()
                    first_step_return = metric()
            final_score_sample = metric()
            final_state = self.save_positions()
            self.load_positions(before_sampling)
            first_time = False
            actions = tuple(actions)
            if actions not in sampled:
                unique = True
        return final_score_sample, actions, final_state, first_step_return, first_step_end_state

    def sample_best(self, num_sample, sample_func):
        best_result = 0
        best_push = None
        best_state = None
        best_first_step = None
        best_first_step_end_state = None
        sampled = set()
        for _ in range(num_sample):
            sample_env = self.copy()
            result, action, state, first_step_return, first_step_end_state = sample_func(sample_env, sampled)
            assert action not in sampled
            sampled.add(action)
            if result > best_result:
                best_result = result
                best_push = action
                best_state = state
                best_first_step = first_step_return
                best_first_step_end_state = first_step_end_state
        return best_result, best_push, best_state, best_first_step, best_first_step_end_state

    def sample_closed_loop(self, num_steps, num_sample, sample_func):
        for i in range(num_steps):
            if num_steps - i == 1:
                best_performance, best_push, best_state = self.greedy_step(prune_method=no_prune, metric=self.count_soft_threshold, sample_size=num_sample)
                first_step_return = best_performance
                first_step_end_state = best_state
            else:
                best_performance, best_push, best_state, first_step_return, first_step_end_state = self.sample_best(num_sample=num_sample, sample_func=sample_func)
                best_push =  (np.array([best_push[0], best_push[1]]), np.array([best_push[2], best_push[3]]))
            self.push(best_push)
            np.testing.assert_allclose(first_step_end_state, self.save_positions(), err_msg="%s \n %s" % (first_step_end_state, self.save_positions()))
            np.testing.assert_almost_equal(first_step_return, self.count_soft_threshold())
            # if not isclose(first_step_return, self.count_soft_threshold(), abs_tol=1e-3):
            #     print("CLOSED LOOP NOT REPRODUCIBLE at step %d. Expected: %s; actual: %s" % (i, first_step_return, self.count_soft_threshold()))
            #     print(first_step_end_state)
            #     print()
            #     print(self.save_positions())
        return best_performance

    def save(self, path=None):
        info = np.hstack([object.save() for object in self.objects] + [len(self.objects)])
        if path is not None:
            np.save(path, info)
        return info

    def save_positions(self, path=None):
        """Save information about current state in a dictionary in sum_path/env.json"""
        lst = [[obj.body.position[0], obj.body.position[1], obj.body.angle] for obj in self.objects]
        if self.rod:
            lst.append([self.rod.position[0], self.rod.position[1], self.rod.angle])
        summary = np.array(lst)
        if path is not None:
            np.save(path, summary)
        return summary

    def snapshot(self, path):
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


def visualize_push(summary_folder, img_folder, save_frames=False):
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)
    env = State(summary=np.load(os.path.join(summary_folder, "i.npy")))
    env.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.iconify()
    images = []
    for timestamp in range(100):
        env.load_positions(np.load(os.path.join(summary_folder, "%d.npy"%timestamp)))
        env.screen.fill(WHITE)
        for obj in env.objects:
            obj.fixture.shape.draw(obj.body, obj.color, env.screen)
        env.rodfix.shape.draw(env.rod, (0, 0, 0, 255), env.screen)
        img_path = os.path.join(img_folder, '%d.png'%timestamp)
        pygame.image.save(env.screen, img_path)
        images.append(imageio.imread(img_path))
        if not save_frames:
            os.remove(img_path)
    pygame.display.quit()
    pygame.quit()
    imageio.mimsave(os.path.join(img_folder, 'push.gif'), images)


if __name__ == "__main__":
    NUM_STEPS = 3
    env = State()
    env.create_random_env(num_objs=2)
    env.sample_closed_loop(num_steps=NUM_STEPS, num_sample=1, sample_func=lambda sample_env, sampled: sample_env.sample(num_steps=NUM_STEPS, prune_method=no_prune, metric=sample_env.count_soft_threshold, sampled=sampled, display=False, save_summary=False, path=None))
