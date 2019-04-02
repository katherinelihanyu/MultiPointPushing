import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
import singulation_env_simulation
import os
import random

from helper import *
from policies import *
from prune import *


def compare_dict(small, big):
    diffkeys = [k for k in small if small[k] != big[k]]
    assert len(diffkeys) > 0
    for k in diffkeys:
        print(k, np.abs(np.array(small[k]) - np.array(big[k])))
        # print(k, ':', small[k], '->', big[k])
    print()


def compare_soft_threshold(env1, env2):
    if len(env1.objs) != len(env2.objs):
        print("length of objects not the same")
        return False
    for i in range(len(env1.objs)):
        if env1.objs[i].fixtures[0].shape.vertices != env2.objs[i].fixtures[0].shape.vertices:
            print("env1.objs[%d].fixtures[0].shape.vertices"%i, env1.objs[i].fixtures[0].shape.vertices)
            print("env2.objs[%d].fixtures[0].shape.vertices"%i, env2.objs[i].fixtures[0].shape.vertices)
            return False
        if env1.objs[i].body.position != env2.objs[i].body.position:
            print("env1.objs[%d].body.position"%i, env1.objs[i].body.position)
            print("env2.objs[%d].body.position"%i, env2.objs[i].body.position)
            return False
        if env1.objs[i].body.angle != env2.objs[i].body.angle:
            print("env1.objs[%d].body.angle"%i, env1.objs[i].body.angle)
            print("env2.objs[%d].body.angle"%i, env2.objs[i].body.angle)
            return False
    return True


def create_initial_envs(num_trials, num_objects, data_path):
    for i in range(num_trials):
        path = os.path.join(data_path, "env" + str(i))
        if not os.path.exists(path):
            os.makedirs(path)
        env = singulation_env_simulation.SingulationEnv()
        env.create_random_env_wrapper(num_objects)
        env.save_env(sum_path=path)


def collect_sequential_sample(complete_pt_lst, summary, model, reuse, max_step, sample_path, metric="count soft threshold", open_loop=False):
    actions = []
    if reuse:
        with open(os.path.join(sample_path, "push_summary.json"), "r") as read_file:
            result = json.load(read_file)
        if open_loop:
            actions.append([result["first push start pt"], result["first push end pt"]])
            actions.append([result["second push start pt"], result["second push end pt"]])
            actions.append([result["third push start pt"], result["third push end pt"]])
        else:
            actions.append([result["first push start pt"], result["first push end pt"]])
    else:
        info = {}
        result = {}
        env = singulation_env_simulation.SingulationEnv()
        env.load_env(summary)
        assert compare_soft_threshold(model, env)
        info["env before push"] = env.save_env()
        for i in range(max_step):
            best_pts = random.choice(complete_pt_lst)
            info["action" + str(i)] = best_pts
            actions.append(best_pts)
            curr_sum = env.collect_data_summary(best_pts[0], best_pts[1], summary, first_step = i==0 and max_step < 3)
            count_after = env.count_soft_threshold()
            info[metric + " before push" + str(i)] = curr_sum[metric + " before push"]
            result[metric + " before push"] = curr_sum[metric + " before push"]
            info[metric + " after push" + str(i)] = curr_sum[metric + " after push"]
            info["env after push" + str(i)] = env.save_env()
            if count_after != curr_sum[metric + " after push"]:
                print("count after not equal", count_after, curr_sum[metric + " after push"])
            step_path = os.path.join(sample_path, "sample_step" + str(i))
            if not os.path.exists(step_path):
                os.makedirs(step_path)
            with open(os.path.join(step_path, "summary.json"), 'w') as f:
                json.dump(curr_sum, f)
        result[metric + " after push"] = curr_sum[metric +" after push"]
        result["first push start pt"] = actions[0][0].tolist()
        result["first push end pt"] = actions[0][1].tolist()
        if len(actions) >1:
            result["second push start pt"] = actions[1][0].tolist()
            result["second push end pt"] = actions[1][1].tolist()
        if len(actions)> 2:
            result["third push start pt"] = actions[2][0].tolist()
            result["third push end pt"] = actions[2][1].tolist()
        with open(os.path.join(sample_path, "push_summary.json"), 'w') as f:
            json.dump(result, f)
    if open_loop:
        return result[metric +" after push"], actions, info
    else:
        return result[metric + " after push"], actions[0], info


def sampling_every_step(summary, data_path, num_samples, num_steps, reuse, reuse_path=None, metric="count soft threshold", timeit=False):
    reuse_step_path = None
    env = singulation_env_simulation.SingulationEnv()
    env.load_env(summary)
    for i in range(num_steps):
        actual_step_path = os.path.join(data_path, "actual_step"+str(i))
        if not os.path.exists(actual_step_path):
            os.makedirs(actual_step_path)
        if reuse:
            reuse_step_path = os.path.join(reuse_path, "actual_step"+str(i))
            if not os.path.exists(reuse_step_path):
                os.makedirs(reuse_step_path)
        # actual_before = env.save_env()
        best_result, best_pts, info = env.best_sequential_sample(num_samples, no_prune, reuse=reuse, max_step=num_steps-i,
                                                           data_path=actual_step_path, reuse_path=reuse_step_path, metric=metric)
        if reuse:
            best_summary = env.collect_data_summary(best_pts[0], best_pts[1],
                                                    img_path=os.path.join(reuse_step_path, "render"), display=False,
                                                    sum_path=reuse_step_path)
        else:
            assert info["env before push"] == env.save_env()
            assert info["action"] == best_pts
            best_summary = env.collect_data_summary(best_pts[0], best_pts[1], img_path=os.path.join(actual_step_path, "render"), display=False, sum_path=actual_step_path)
            assert info[metric + " before push"] == best_summary[metric + " before push"], "before: step: %r; predicted: %r; actual: %r" % (i, info[metric + " before push"], best_summary[metric + " before push"])
            if info[metric + " after push"] != best_summary[metric + " after push"]:
                print("step", i)
                assert env.save_env() != info["env after push"]
#                 print("predicted", info[metric + " after push"])
#                 print("actual", best_summary[metric + " after push"])
                compare_dict(info["env after push"], env.save_env())
#         if timeit:
#             print("sampling step %d took"%i, datetime.datetime.now().replace(microsecond=0) - cur_time)
#             cur_time = datetime.datetime.now().replace(microsecond=0)
#     if timeit:
#         print("sampling took", datetime.datetime.now().replace(microsecond=0) - first_time)
    return best_summary[metric + " after push"]


def sampling_open_loop(summary, data_path, num_samples, num_steps, reuse, metric="count soft threshold", timeit=False):
    first_time = datetime.datetime.now().replace(microsecond=0)
    cur_time = datetime.datetime.now().replace(microsecond=0)
    env = singulation_env_simulation.SingulationEnv()
    env.load_env(summary)
    actual_step_path = os.path.join(data_path, "samples")
    if not os.path.exists(actual_step_path):
        os.makedirs(actual_step_path)
    best_result, actions, info = env.best_sequential_sample(num_samples, no_prune, reuse=reuse, max_step=num_steps,
                                                            data_path=actual_step_path, metric=metric, open_loop=True)
    for i in range(len(actions)):
        best_pts = actions[i]
        print("step", i)
        actual_step_path = os.path.join(data_path, "actual_step" + str(i))
        if not os.path.exists(actual_step_path):
            os.makedirs(actual_step_path)
        best_summary = env.collect_data_summary(best_pts[0], best_pts[1], img_path=actual_step_path, display=False, sum_path=actual_step_path)
        assert best_pts == info["action" + str(i)]
        # print("predicted action", info["action"])
        # print("actual action", best_pts)
        # print("predicted", metric, "before push", info[metric + " before push"])
        # print("actual", metric, "before push", best_summary[metric + " before push"])
        assert info[metric + " before push" + str(i)] == best_summary[metric + " before push"], "Before: predicted: %r; actual: %r" % (info[metric + " before push" + str(i)], best_summary[metric + " before push"])
        # print("predicted", metric, "after push", info[metric + " after push"])
        # print("actual", metric, "after push", best_summary[metric + " after push"])
        assert info[metric + " after push" + str(i)] == best_summary[metric + " after push"], "After: predicted: %r; actual: %r" % (info[metric + " after push" + str(i)], best_summary[metric + " after push"])
        if timeit:
            print("sampling step %d took"%i, datetime.datetime.now().replace(microsecond=0) - cur_time)
            cur_time = datetime.datetime.now().replace(microsecond=0)
    if timeit:
        print("sampling took", datetime.datetime.now().replace(microsecond=0) - first_time)
    return best_summary[metric + " after push"]


def sampling_last_step_greedy(summary, data_path, num_samples, num_steps, reuse, metric="count soft threshold", timeit=False):
    first_time = datetime.datetime.now().replace(microsecond=0)
    cur_time = datetime.datetime.now().replace(microsecond=0)
    env = singulation_env_simulation.SingulationEnv()
    env.load_env(summary)
    for i in range(num_steps-1):
        print("step", i)
        actual_step_path = os.path.join(data_path, "actual_step"+str(i))
        if not os.path.exists(actual_step_path):
            os.makedirs(actual_step_path)
        best_result, best_pts = env.best_sequential_sample(num_samples, no_prune, reuse=reuse, max_step=num_steps-i,
                                                           data_path=actual_step_path, metric=metric,)
        best_summary = env.collect_data_summary(best_pts[0], best_pts[1], img_path=os.path.join(actual_step_path, "render"), display=True)
        print(metric + " before push", best_summary[metric + " before push"], "projected reward at step %d: %.2f" % (i + 3, best_result))
        print(metric + " after push", best_summary[metric + " after push"])
        if timeit:
            print("sampling step %d took"%i, datetime.datetime.now().replace(microsecond=0) - cur_time)
            cur_time = datetime.datetime.now().replace(microsecond=0)
    print("step", num_steps-1)
    curr_pos = env.save_curr_position()
    actual_step_path = os.path.join(data_path, "actual_step" + str(num_steps-1))
    if not os.path.exists(actual_step_path):
        os.makedirs(actual_step_path)
    best_pts = env.prune_best(prune_method=no_prune, metric=metric, position=curr_pos)
    best_summary = env.collect_data_summary(best_pts[0], best_pts[1], img_path=os.path.join(actual_step_path, "render"), display=True)
    print(metric + " before push", best_summary[metric + " before push"])
    print(metric + " after push", best_summary[metric + " after push"])
    if timeit:
        print("sampling step %d took" % (num_steps-1), datetime.datetime.now().replace(microsecond=0) - cur_time)
        print("sampling took", datetime.datetime.now().replace(microsecond=0) - first_time)
    return best_summary[metric + " after push"]


def greedy_sequential(summary, num_steps, data_path, metric="count soft threshold", display=False, num_samples=None, reuse=False, timeit=False):
    cur_time = time.time()
    test = singulation_env_simulation.SingulationEnv()
    test.load_env(summary)
    test.visualize("./1.png")
    for i in range(num_steps):
        print("step", i)
        actual_step_path = os.path.join(data_path, "step"+str(i))
        if not os.path.exists(actual_step_path):
            os.makedirs(actual_step_path)
        if not os.path.exists(os.path.join(actual_step_path, "render")):
            os.makedirs(os.path.join(actual_step_path, "render"))
        if not os.path.exists(os.path.join(actual_step_path, "summary")):
            os.makedirs(os.path.join(actual_step_path, "summary"))
        curr_pos = test.save_curr_position()
        curr_env = test.save_env()
        best_pts, predicted_summary = test.prune_best(prune_method=no_prune, metric=metric, position=curr_pos)
        assert test.save_env() == curr_env
        if best_pts is None:
            print("best_pts is None")
            break
        if display:
            best_summary = test.collect_data_summary(
                best_pts[0], best_pts[1], img_path=os.path.join(actual_step_path, "render"), display=display,
                sum_path=os.path.join(actual_step_path, "summary"))
        else:
            best_summary = test.collect_data_summary(best_pts[0], best_pts[1])
        test.visualize("./2.png")
        best_dist = best_summary[metric + " after push"] - best_summary[metric + " before push"]
        assert predicted_summary[metric + " before push"] == best_summary[metric + " before push"], "before: step: %r; predicted: %r; actual: %r" % (i, predicted_summary[metric + " before push"], best_summary[metric + " before push"])
        # assert predicted_summary[metric + " after push"] == best_summary[metric + " after push"], "after: step: %r; predicted: %r; actual: %r" % (i, predicted_summary[metric + " after push"], best_summary[metric + " after push"])
        # print("predicted", metric, "before push", predicted_summary[metric + " before push"])
        # print("actual", metric, "before push", best_summary[metric + " before push"])
        if predicted_summary[metric + " after push"] != best_summary[metric + " after push"]:
            # print("predicted", metric, "after push", predicted_summary[metric + " after push"])
            # print("actual", metric, "after push", best_summary[metric + " after push"])
            print(metric, "after push", np.abs(predicted_summary[metric + " after push"] - best_summary[metric + " after push"]))
            compare_dict(predicted_summary["env after push"], test.save_env())
        print("greedy step %d took"%i, time.time()-cur_time)
        cur_time = time.time()
        if best_dist <= 0:
            print("best_dist <= 0")
            return best_summary[metric + " before push"]
    return best_summary[metric + " after push"]


def run_experiments(num_trials,data_path, func, reuse):
    lst = []
    for i in range(num_trials):
        result = run_heap(data_path, i, func, reuse)
        lst.append(result)
    return lst


def run_heap(data_path, heap_num, func, reuse):
    print("heap", heap_num)
    path_i = os.path.join(data_path, "env" + str(heap_num))
    with open(os.path.join(path_i, "env.json"), "r") as read_file:
        summary = json.load(read_file)
    result = func(summary, path_i, reuse)
    return result


def plot(num_samples_lst, means, stds, num_objects,num_trials,num_steps,path):
    plt.figure()
    plt.errorbar(num_samples_lst, means, yerr=stds)
    plt.title("%d-push sampling with %d objects and %d trials"%(num_steps,num_objects,num_trials))
    plt.xlabel('Number of samples')
    plt.ylabel('Count threshold')
    plt.savefig(path)

if __name__ == "__main__":
    num_objects = 10
    num_trials = 1
    num_steps = 1
    beg_time = datetime.datetime.now().replace(microsecond=0)
    num_samples = 30
    mypath = "/nfs/diskstation/katherineli/sampling1"

    returns = run_experiments(
        num_trials, data_path=mypath, reuse=False, func=lambda summary, data_path, reuse: greedy_sequential(
            summary=summary, data_path=data_path, num_samples=num_samples,
            num_steps=num_steps, reuse=reuse, metric="count soft threshold", timeit=True))
    m = np.mean(returns)
    s = np.std(returns)
    print("%d samples, mean: %.2f, std: %.2f"%(num_samples, m, s))
    time_elapsed = datetime.datetime.now().replace(microsecond=0) - beg_time
    print("Time elapsed:", time_elapsed)
    with open(os.path.join(mypath, '200samples_returns.pickle'), 'wb') as f:
        pickle.dump(returns, f)
    with open(os.path.join(mypath, '200samples_time.pickle'), 'wb') as f:
        pickle.dump(time_elapsed, f)
