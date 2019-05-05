import numpy as np


def run_heap(data_path, heap_num, num_steps):
    path_i = os.path.join(data_path, "env%d.npy" % heap_num)
    info = np.load(path_i)
    num_objects = int(info[-1])
    print("num_objects", num_objects)
    env1 = State(summary=info)
    before_summary = env1.save_positions()
    env2 = env1.copy()
    np.testing.assert_array_equal(before_summary, env2.save_positions())
    final_score, actions, final_state, first_step_return, first_step_end_state = \
        env1.sample(num_steps=NUM_STEPS, prune_method=no_prune, metric=env1.count_soft_threshold, sampled=set())
    np.testing.assert_array_equal(before_summary, env1.save_positions())
    np.testing.assert_array_equal(before_summary, env2.save_positions())
    for i in range(num_steps):
        action = (np.array([actions[i * 4], actions[i * 4 + 1]]), np.array([actions[i * 4 + 2], actions[i * 4 + 3]]))
        env2.push(action)
        if i == 0:
            np.testing.assert_array_equal(first_step_end_state, env2.save_positions(),
                                          err_msg="Iteration %d First step \n %s \n %s" % (
                                              it, first_step_end_state, env2.save_positions()))
            assert first_step_return == env2.count_soft_threshold()
    np.testing.assert_array_equal(final_state, env2.save_positions(),
                                  err_msg="iteration %d Final \n %s \n %s" % (it, final_state, env2.save_positions()))
    assert final_score == env2.count_soft_threshold()
    print("heap%d: %s" % (heap_num, result))
    return result


if __name__ == '__main__':
    result = run_heap(data_path="/nfs/diskstation/katherineli/states/1_obj", heap_num=0, num_steps=3)
    print("result", result)