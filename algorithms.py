import numpy as np
import os
from state import *


def create_initial_envs(start_index, end_index, num_objects, data_path):
    """Create initial environments in data_path numbered from start_index to end_index.
    """
    for i in range(start_index, end_index):
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        path = os.path.join(data_path, "env%d.npy" % i)
        env = State()
        env.create_random_env(num_objs=num_objects)
        env.save(path=path)


def run_experiments(num_heaps, data_path, num_steps, num_sample, algorithm):
    lst = []
    for i in range(num_heaps):
        result = run_heap(data_path, i, num_steps, num_sample, algorithm)
        print("result", result)
        lst.append(result)
        print(lst)
    return lst


def run_heap(data_path, heap_num, num_steps, num_sample, algorithm):
    path_i = os.path.join(data_path, "env%d.npy" % heap_num)
    info = np.load(path_i)
    num_objects = int(info[-1])
    print("num_objects", num_objects)
    env = State(summary=info)
    if algorithm == "greedy":
        result, actions = env.greedy(num_steps=num_steps, prune_method=no_prune, metric=env.count_soft_threshold)
    elif algorithm == "open_loop":
        result, best_push, best_state, best_first_step, best_first_step_end_state = env.sample_best(num_sample=num_sample, sample_func=lambda sample_env, sampled: sample_env.sample(num_steps=num_steps, prune_method=no_prune, metric=sample_env.count_soft_threshold, sampled=sampled))
    elif algorithm == "closed_loop":
        result = env.sample_closed_loop(num_steps=num_steps, num_sample=num_sample, sample_func=lambda sample_env, sampled, step, num_steps: sample_env.sample(num_steps=num_steps, prune_method=no_prune, metric=sample_env.count_soft_threshold, sampled=sampled, display=False, path="sample_step%d_"%step))
    else:
        raise ValueError("algorithm is not greedy, open_loop or closed_loop")
    print("heap%d: %s" % (heap_num, result))
    return result


if __name__ == "__main__":
    algorithms = ["greedy", "open_loop", "closed_loop"]
    
    # hyperparameters
    num_heaps = 50
    data_path = "/nfs/diskstation/katherineli/states/10_objs" # path to the starting states
    num_steps = 3
    num_sample = 200
    algorithm = "closed_loop"
    
    returns = run_experiments(num_heaps=num_heaps, data_path=data_path, num_steps=num_steps, num_sample=num_sample, algorithm=algorithm)
    m = np.mean(returns)
    s = np.std(returns)
    print("mean: %.2f, std: %.2f"%(m, s))
    print(returns)
