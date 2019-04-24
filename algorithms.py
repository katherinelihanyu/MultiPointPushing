import numpy as np
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


def run_experiments(num_heaps, data_path):
    lst = []
    for i in range(num_heaps):
        result = run_heap(data_path, i)
        lst.append(result)
    return lst


def run_heap(data_path, heap_num):
    path_i =  os.path.join(data_path, "env%d.npy" % heap_num)
    info = np.load(path_i)
    num_objects = int(info[-1])
    env = State(summary=info, num_objs=num_objects)
    # result, actions = env.greedy(num_steps=3, prune_method=no_prune, metric=env.count_soft_threshold)
    result, actions, state = env.sample_best(num_sample=200, sample_func=lambda sample_env, sampled: sample_env.sample(
        num_steps=3, prune_method=no_prune, metric=sample_env.count_soft_threshold, sampled=sampled))
    print("heap%d: %s" % (heap_num, result))
    return result


if __name__ == "__main__":
    returns = run_experiments(num_heaps=50, data_path="/nfs/diskstation/katherineli/states")
    m = np.mean(returns)
    s = np.std(returns)
    print("mean: %.2f, std: %.2f"%(m, s))
    print(returns)
