import numpy as np
import os
from state import *


def run_heap2(data_path, heap_num, num_steps):
    if num_steps == 1:
        return run_heap_one_step(data_path, heap_num)
    path_i = os.path.join(data_path, "env%d.npy" % heap_num)
    info = np.load(path_i)
    num_objects = int(info[-1])
    env1 = State(summary=info)
    before_summary = env1.save_positions()
    env2 = env1.copy()
    np.testing.assert_array_equal(before_summary, env2.save_positions())
    final_score, actions, final_state, first_step_return, first_step_end_state = env1.sample_best(num_sample=100, sample_func=lambda sample_env, sampled: sample_env.sample(num_steps=num_steps, prune_method=no_prune, metric=sample_env.count_soft_threshold, sampled=sampled))
    np.testing.assert_array_equal(before_summary, env1.save_positions())
    np.testing.assert_array_equal(before_summary, env2.save_positions())
    for i in range(num_steps):
        action = (np.array([actions[i * 4], actions[i * 4 + 1]]), np.array([actions[i * 4 + 2], actions[i * 4 + 3]]))
        env2.push(action)
        if i == 0:
            np.testing.assert_array_equal(first_step_end_state, env2.save_positions(),
                                          err_msg="First step \n %s \n %s" % (first_step_end_state, env2.save_positions()))
            assert first_step_return == env2.count_soft_threshold()
    np.testing.assert_array_equal(final_state, env2.save_positions(),
                                  err_msg="Final \n %s \n %s" % (final_state, env2.save_positions()))
    assert final_score == env2.count_soft_threshold()
    return final_score

def run_heap_one_step2(data_path, heap_num):
    path_i = os.path.join(data_path, "env%d.npy" % heap_num)
    info = np.load(path_i)
    num_objects = int(info[-1])
    env1 = State(summary=info)
    before_summary = env1.save_positions()
    env2 = env1.copy()
    np.testing.assert_array_equal(before_summary, env2.save_positions())
    best_performance, action, best_state = env1.greedy_step(prune_method=no_prune, metric=env1.count_soft_threshold, sample_size=100)
    np.testing.assert_array_equal(before_summary, env1.save_positions())
    np.testing.assert_array_equal(before_summary, env2.save_positions())
    env2.push(action)
    np.testing.assert_array_equal(best_state, env2.save_positions(),
                                          err_msg="\n %s \n %s" % (best_state, env2.save_positions()))
    assert best_performance == env2.count_soft_threshold()
    return best_performance

def run_heap(data_path, heap_num, num_steps):
    if num_steps == 1:
        return run_heap_one_step(data_path, heap_num)
    path_i = os.path.join(data_path, "env%d.npy" % heap_num)
    info = np.load(path_i)
    num_objects = int(info[-1])
    max_pos_diff = 0
    max_diff_step = -1
    max_count_diff = 0
    env1 = State(summary=info)
    before_summary = env1.save_positions()
    env2 = env1.copy()
    np.testing.assert_array_equal(before_summary, env2.save_positions())
    final_score, actions, final_state, first_step_return, first_step_end_state = env1.sample_best(num_sample=500, sample_func=lambda sample_env, sampled: sample_env.sample(num_steps=num_steps, prune_method=no_prune, metric=sample_env.count_soft_threshold, sampled=sampled))
    np.testing.assert_array_equal(before_summary, env1.save_positions())
    np.testing.assert_array_equal(before_summary, env2.save_positions())
    for i in range(num_steps):
        action = (np.array([actions[i * 4], actions[i * 4 + 1]]), np.array([actions[i * 4 + 2], actions[i * 4 + 3]]))
        env2.push(action)
        if i == 0:
            pos_diff = np.max(np.abs(first_step_end_state - env2.save_positions()))
            if pos_diff > max_pos_diff:
                max_pos_diff = pos_diff
                max_diff_step = i
                assert np.abs(first_step_return - env2.count_soft_threshold()) >= max_count_diff
    pos_diff = np.max(np.abs(final_state - env2.save_positions()))
    if pos_diff > max_pos_diff:
            max_pos_diff = pos_diff
            max_diff_step = i
            assert np.abs(final_score - env2.count_soft_threshold()) >= max_count_diff
    return final_score, max_pos_diff, max_count_diff, max_diff_step

def run_heap_one_step(data_path, heap_num):
    path_i = os.path.join(data_path, "env%d.npy" % heap_num)
    info = np.load(path_i)
    num_objects = int(info[-1])
    env1 = State(summary=info)
    before_summary = env1.save_positions()
    env2 = env1.copy()
    np.testing.assert_array_equal(before_summary, env2.save_positions())
    best_performance, action, best_state = env1.greedy_step(prune_method=no_prune, metric=env1.count_soft_threshold, sample_size=500)
    np.testing.assert_array_equal(before_summary, env1.save_positions())
    np.testing.assert_array_equal(before_summary, env2.save_positions())
    env2.push(action)
    pos_diff = np.max(np.abs(best_state - env2.save_positions()))
    count_diff = np.abs(best_performance - env2.count_soft_threshold())
    return best_performance, pos_diff, count_diff, 0

if __name__ == '__main__':
    for num_objects in range(12, 14):
        for num_pushes in range(1, 4):
            for i in range(10):
                result, pos_diff, count_diff, max_diff_step = run_heap(data_path="/nfs/diskstation/katherineli/states/%d_objs" % num_objects, heap_num=0, num_steps=num_pushes)
                if pos_diff == 0 and count_diff == 0:
                    print("num_objects: %d, num_pushes: %d, iteration: %d, result: %s" % (num_objects, num_pushes, i, result))
                else:
                    print("num_objects: %d, num_pushes: %d, iteration: %d, result: %s, pos_diff: %s, count_diff: %s, step: %d ERROR" % (num_objects, num_pushes, i, result, pos_diff, count_diff, max_diff_step))