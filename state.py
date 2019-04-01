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
WORLD = b2World(gravity=(0, 0), doSleep=True)
FRICTION = 0.5
# 'POLYGON_SHAPES' is a list of (minimum radius, maximum radius, number of vertices) tuples.
POLYGON_SHAPES = [(1, 1, 15), (0.75, 1, 12), (0.5, 1, 9), (0.25, 1, 6)]


class Polygon:
    def __init__(self, position=None, vertices=None, shape=0, color=None):
        if vertices is None:
            self.vertices = polygon(*POLYGON_SHAPES[shape])
        else:
            self.vertices = vertices
        if position is None:
            self.position = np.random.uniform(low=4, high=8, size=2)
        else:
            self.position = position
        if color is None:
            self.color = random.choice(COLORS)
        else:
            self.color = color
        self.body = WORLD.CreateDynamicBody(position=self.position.tolist(), allowSleep=False)
        self.fixture = self.body.CreatePolygonFixture(density=1, vertices=self.vertices, friction=FRICTION)
    
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
            obj.body.DestroyFixture(obj.fixture)
            WORLD.DestroyBody(obj.body)
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
    num_env = 5
    for i in range(num_env):
        env = State()
        env.create_random_env(num_objs=5)
        env.visualize("./%d.png"%i)
        print("env%d:"%i, env.count_soft_threshold())