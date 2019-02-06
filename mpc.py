import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pickle

from helper import *
from policies import *
from prune import *
from singulation_env_simulation import *

def sampling(test, num_samples, num_steps):
    # cur_time = time.time()
    for i in range(num_steps):
        # print("step", i)
        best_result, best_pts = test.best_sequential_sample(num_samples, no_prune, max_step=3, metric="count threshold")
        # best_summary = test.collect_data_summary(best_pts[0], best_pts[1], img_path="./data/sequential_sampling_step%d"%i, display=True)
        # best_dist = best_summary["count threshold after push"] - best_summary["count threshold before push"]
        # print("count threshold after push", best_summary["count threshold after push"])
        # print("count threshold before push", best_summary["count threshold before push"])
        # print("step %d took"%i, time.time()-cur_time)
        # cur_time = time.time()
    return best_result

def greedy_sequential(test, num_steps):
    cur_time = time.time()
    for i in range(num_steps):
        print("step", i)
        curr_pos = test.save_curr_position()
        best_pts = test.prune_best(prune_method=no_prune, metric="count threshold", position=curr_pos)
        if best_pts is None:
            print("best_pts is None")
            break
        best_summary = test.collect_data_summary(best_pts[0], best_pts[1], img_path="./data/sequential_greedy_%d"%i, display=True)
        # best_summary = test.collect_data_summary(best_pts[0], best_pts[1])
        best_dist = best_summary["count threshold after push"] - best_summary["count threshold before push"]
        print("count threshold after push", best_summary["count threshold after push"])
        print("count threshold before push", best_summary["count threshold before push"])
        print("step %d took"%i, time.time()-cur_time)
        cur_time = time.time()
        if best_dist <= 0:
            print("best_dist <= 0")
            return best_summary["count threshold before push"]
    return best_summary["count threshold after push"]

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

# avg, lst = get_avg(3,lambda test: greedy_sequential(test,num_steps))

num_objects = 10
num_trials = 50
num_steps = 3
num_samples_lst = [100, 150, 300, 450, 600, 750, 900]
returns = []
means = []
stds = []
cur_time = datetime.datetime.now().replace(microsecond=0)
for num_samples in num_samples_lst:
    returns = get_avg(num_trials, num_objects, lambda test: sampling(test, num_samples, num_steps))
    m = np.mean(returns)
    s = np.std(returns)
    print("%d samples, mean: %.2f, std: %.2f"%(num_samples, m, s))
    print("Time elapsed:", datetime.datetime.now().replace(microsecond=0) - cur_time)
    cur_time = datetime.datetime.now().replace(microsecond=0)
    means.append(m)
    stds.append(s)

with open('./data/data.pkl', 'wb') as f:
    pickle.dump([means, stds], f)

plt.figure()
plt.errorbar(num_samples_lst, means, yerr=stds)
plt.xlabel('Number of samples')
plt.ylabel('Count threshold')
plt.savefig("./data/graph.png")