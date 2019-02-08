import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pickle

from helper import *
from policies import *
from prune import *
from singulation_env_simulation import *

def sampling(test, num_samples, num_steps, metric="count soft threshold", timeit=False):
    cur_time = time.time()
    for i in range(num_steps):
        best_result, best_pts = test.best_sequential_sample(num_samples, no_prune, max_step=3, metric=metric)
        best_summary = test.collect_data_summary(best_pts[0], best_pts[1], img_path="./data/sequential_sampling_step%d"%i, display=True)
        print(metric + " after push", best_summary[metric + " after push"])
        print(metric + " before push", best_summary[metric + " before push"])
        if timeit:
            print("sampling step %d took"%i, time.time()-cur_time)
            print("projected reward at step %d: %.2f"%(i+3, best_result))
            cur_time = time.time()
    return best_summary[metric + " after push"]

def greedy_sequential(test, num_steps, metric="count soft threshold", save_img = False):
    cur_time = time.time()
    for i in range(num_steps):
        print("step", i)
        curr_pos = test.save_curr_position()
        best_pts = test.prune_best(prune_method=no_prune, metric=metric, position=curr_pos)
        if best_pts is None:
            print("best_pts is None")
            break
        if save_img:
            best_summary = test.collect_data_summary(best_pts[0], best_pts[1], img_path="./data/sequential_greedy_%d"%i, display=True)
        else:
            best_summary = test.collect_data_summary(best_pts[0], best_pts[1])
        best_dist = best_summary[metric + " after push"] - best_summary[metric + " before push"]
        print(metric + " after push", best_summary[metric + " after push"])
        print(metric + " before push", best_summary[metric + " before push"])
        print("greedy step %d took"%i, time.time()-cur_time)
        cur_time = time.time()
        if best_dist <= 0:
            print("best_dist <= 0")
            return best_summary[metric + " before push"]
    return best_summary[metric + " after push"]

def get_avg(num_trials, num_objects, func):
    lst = []
    for i in range(num_trials):
        test = SingulationEnv()
        while True:
            try:
                test.create_random_env(num_objects)
                break
            except:
                pass
        result = func(test)
        lst.append(result)
    return lst

def plot(num_samples_lst, means, stds, num_objects,num_trials,num_steps,path):
    plt.figure()
    plt.errorbar(num_samples_lst, means, yerr=stds)
    plt.title("%d-push sampling with %d objects and %d trials"%(num_steps,num_objects,num_trials))
    plt.xlabel('Number of samples')
    plt.ylabel('Count threshold')
    plt.savefig(path)


num_objects = 10
num_trials = 20
num_steps = 3

test = SingulationEnv()
test.create_random_env(num_objects)
summary = test.save_env()
best_result = sampling(test, num_samples=num_samples, num_steps=num_steps, metric="count soft threshold", timeit=True)
print("sampling with %d samples, result:"%num_samples, best_result)
test = SingulationEnv()
test.load_env(summary)
best_result = greedy_sequential(test=test, num_steps=num_steps, metric="count soft threshold", save_img = True)
print("greedy result:", best_result)

# num_samples_lst = [10, 150, 300, 450, 600, 750, 900, 1050, 1200]
# returns = []
# means = []
# stds = []
# cur_time = datetime.datetime.now().replace(microsecond=0)
# for num_samples in num_samples_lst:
#     returns = get_avg(num_trials, num_objects, lambda test: sampling(test, num_samples, num_steps, "count soft threshold"))
#     m = np.mean(returns)
#     s = np.std(returns)
#     print("%d samples, mean: %.2f, std: %.2f"%(num_samples, m, s))
#     print("Time elapsed:", datetime.datetime.now().replace(microsecond=0) - cur_time)
#     cur_time = datetime.datetime.now().replace(microsecond=0)
#     means.append(m)
#     stds.append(s)
#
# with open('./data/data.pkl', 'wb') as f:
#     pickle.dump([means, stds], f)
# #
# means = [5.52,6.42,6.75,6.8,6.9,7.19, 6.95, 6.93, 7.14]
# stds = [0.97,1.04,0.79,0.81,1.1,1.2,0.99,0.93, 0.97]