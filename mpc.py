import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json

from helper import *
from policies import *
from prune import *
from singulation_env_simulation import *

def compare_dict(small, big):
    diffkeys = [k for k in small if small[k] != big[k]]
    assert len(diffkeys) > 0
    for k in diffkeys:
        print(k, np.abs(np.array(small[k]) - np.array(big[k])))
        # print(k, ':', small[k], '->', big[k])
    print()

def sampling_every_step(summary, data_path, num_samples, num_steps, reuse, reuse_path=None, metric="count soft threshold", timeit=False):
    first_time = datetime.datetime.now().replace(microsecond=0)
    cur_time = datetime.datetime.now().replace(microsecond=0)
    reuse_step_path = None
    env = SingulationEnv()
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
    env = SingulationEnv()
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
        print("predicted action", info[i*3])
        print("actual action", best_pts)
        print("predicted", metric, "before push", info[i*3+1])
        print("actual", metric, "before push", best_summary[metric + " before push"])
        print("predicted", metric, "after push", info[3*i+2])
        print("actual", metric, "after push", best_summary[metric + " after push"])
        if timeit:
            print("sampling step %d took"%i, datetime.datetime.now().replace(microsecond=0) - cur_time)
            cur_time = datetime.datetime.now().replace(microsecond=0)
    if timeit:
        print("sampling took", datetime.datetime.now().replace(microsecond=0) - first_time)
    return best_summary[metric + " after push"]

def sampling_last_step_greedy(summary, data_path, num_samples, num_steps, reuse, metric="count soft threshold", timeit=False):
    first_time = datetime.datetime.now().replace(microsecond=0)
    cur_time = datetime.datetime.now().replace(microsecond=0)
    env = SingulationEnv()
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
    test = SingulationEnv()
    test.load_env(summary)
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
        best_pts, predicted_summary = test.prune_best(prune_method=no_prune, metric=metric, position=curr_pos)
        if best_pts is None:
            print("best_pts is None")
            break
        if display:
            best_summary = test.collect_data_summary(
                best_pts[0], best_pts[1], img_path=os.path.join(actual_step_path, "render"), display=display,
                sum_path=os.path.join(actual_step_path, "summary"))
        else:
            best_summary = test.collect_data_summary(best_pts[0], best_pts[1])
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

num_objects = 10
num_trials = 30
num_steps = 1
beg_time = datetime.datetime.now().replace(microsecond=0)
num_samples = 30
mypath = "/nfs/diskstation/katherineli/sampling1"

returns = run_experiments(num_trials, data_path=mypath, reuse=False,
                          func=lambda summary, data_path, reuse: greedy_sequential(summary=summary, data_path=data_path,
                                                                                        num_samples=num_samples,
                                                                                        num_steps=num_steps,
                                                                                        reuse=reuse,
                                                                                        metric="count soft threshold",
                                                                                        timeit=True))
m = np.mean(returns)
s = np.std(returns)
print("%d samples, mean: %.2f, std: %.2f"%(num_samples, m, s))
time_elapsed = datetime.datetime.now().replace(microsecond=0) - beg_time
print("Time elapsed:", time_elapsed)
with open(os.path.join(mypath, '200samples_returns.pickle'), 'wb') as f:
    pickle.dump(returns, f)
with open(os.path.join(mypath, '200samples_time.pickle'), 'wb') as f:
    pickle.dump(time_elapsed, f)
