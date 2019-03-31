import pygame
import matplotlib.pyplot as plt

from Box2D import (b2Distance, b2PolygonShape, b2Transform, b2World)
from helpers import *

"""Display variables."""
SCREEN_WIDTH, SCREEN_HEIGHT = 720, 720
PPM = 60.0  # pixels per meter
WHITE = (255, 255, 255, 255)
num_classes = 8
color_list = plt.cm.get_cmap('Accent')(np.linspace(0, 1, num_classes))
COLORS = [(int(255*c[0]), int(255*c[1]), int(255*c[2]), int(255*c[3])) for c in color_list]

"""Physics variables."""
WORLD = b2World(gravity=(0, 0), doSleep=True)
FRICTION = 0.5
# 'POLYGON_SHAPES' is a list of (minimum radius, maximum radius, number of vertices) tuples.
POLYGON_SHAPES = [(1, 1, 15), (0.75, 1, 12), (0.5, 1, 9), (0.25, 1, 6)]


class Polygon:
    def __init__(self, position=None, vertices=None, shape=None, color=WHITE):
        if vertices is None:
            self.vertices = polygon(*POLYGON_SHAPES[shape])
        else:
            self.vertices = vertices
        if position is None:
            self.position = np.random.uniform(low=4, high=8, size=2)
        else:
            self.position = position
        self.body = WORLD.CreateDynamicBody(position=self.position.tolist(), allowSleep=False)
        self.fixture = self.body.CreatePolygonFixture(density=1, vertices=self.vertices, friction=FRICTION)
        self.color = color
    
    def distance(self, other_polygon):
        shape1 = self.fixture.shape
        shape2 = other_polygon.fixture.shape
        transform1 = b2Transform()
        pos1 = self.body.position
        angle1 = self.body.angle
        transform1.Set(pos1, angle1)
        transform2 = b2Transform()
        pos2 = other_polygon.body.position
        angle2 = other_polygon.body.angle
        transform2.Set(pos2, angle2)
        dist = b2Distance(
            shapeA=shape1, shapeB=shape2, transformA=transform1, transformB=transform2)[2]
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
        pygame.draw.polygon(self.screen, (0, 0, 0, 0), vertices, 5)  # boundary of rectangle

    def clear(self):
        """Remove all objects."""
        for obj in self.objects:
            obj.body.DestroyFixture(obj.fixture)
            WORLD.DestroyBody(obj.body)
        self.objects = []

    def create_random_env(self, num_objs=3, shape=0, max_iter_limit=10000):
        self.clear()
        for i in range(num_objs):
            if len(self.objects) == 0:
                self.objects.append(Polygon(shape=shape, color=COLORS[i]))
            else:
                # import pdb; pdb.set_trace()
                max_iter = max_iter_limit
                while True:
                    if max_iter <= 0:
                        raise RuntimeError("max iter reaches")
                    max_iter -= 1
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
    # data_path = "/nfs/diskstation/katherineli/sampling"
    #  create_initial_envs(50,10,data_path)
    env = State()
    env.create_random_env(num_objs=9)
    env.visualize("./no_norm.png")