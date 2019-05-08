import numpy as np
import os
from state import *


def create_initial_envs(start, end, num_objects, data_path):
    for i in range(start, end):
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
        print(lst)
    return lst


def run_heap(data_path, heap_num):
    path_i = os.path.join(data_path, "env%d.npy" % heap_num)
    info = np.load(path_i)
    num_objects = int(info[-1])
    print("num_objects", num_objects)
    env = State(summary=info)
    # result, actions = env.greedy(num_steps=3, prune_method=no_prune, metric=env.count_soft_threshold)
    result, best_push, best_state, best_first_step, best_first_step_end_state = env.sample_best(num_sample=3400, sample_func=lambda sample_env, sampled: sample_env.sample(
        num_steps=3, prune_method=no_prune, metric=sample_env.count_soft_threshold, sampled=sampled))
#     result = env.sample_closed_loop(num_sample=200, sample_func=lambda sample_env, sampled: sample_env.sample(
#         num_steps=3, prune_method=no_prune, metric=sample_env.count_soft_threshold, sampled=sampled), num_steps=3)
    print("heap%d: %s" % (heap_num, result))
    return result


if __name__ == "__main__":
#     returns = run_experiments(num_heaps=1, data_path="/nfs/diskstation/katherineli/states/1_obj")
#     m = np.mean(returns)
#     s = np.std(returns)
#     print("mean: %.2f, std: %.2f"%(m, s))
#     print(returns)
    for num_object in range(8, 9):
        create_initial_envs(start=46, end=50, num_objects=num_object, data_path="/nfs/diskstation/katherineli/states/%d_objs" % num_object)
