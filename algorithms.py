import os
from state import *


def create_initial_envs(num_heaps, num_objects, data_path):
    for i in range(num_heaps):
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        path = os.path.join(data_path, "env%d.npy" % i)
        env = State()
        env.create_random_env(num_objs=num_objects)
        env.save(path=path)


if __name__ == "__main__":
    create_initial_envs(num_heaps=3, num_objects=10, data_path="states")
