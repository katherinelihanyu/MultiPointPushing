import time

from helper import *
from policies import *
from prune import *
from singulation_env_simulation import *

def sampling(test, num_samples, num_steps):
    cur_time = time.time()
    for i in range(num_steps):
        print("step", i)
        best_result, best_pts = test.best_sequential_sample(num_samples, no_prune, max_step=3, metric="count threshold")
        best_summary = test.collect_data_summary(best_pts[0], best_pts[1], img_path="./data/sequential_sampling_step%d"%i, display=True)
        # best_dist = best_summary["count threshold after push"] - best_summary["count threshold before push"]
        print("count threshold after push", best_summary["count threshold after push"])
        print("count threshold before push", best_summary["count threshold before push"])
        print("step %d took"%i, time.time()-cur_time)
        cur_time = time.time()
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

def get_avg(num_trials, func):
    lst = []
    for i in range(num_trials):
        print("trial", i)
        test = SingulationEnv()
        while True:
            try:
                test.create_random_env(num_objects)
                break
            except:
                pass
        result = func(test)
        lst.append(result)
    avg = sum(lst)/num_trials
    return avg, lst

# avg, lst = get_avg(3,lambda test: greedy_sequential(test,num_steps))

num_objects=10
path="./data"
num_trials = 1
num_steps = 3
num_samples = 100

lst = []
for i in range(num_trials):
    print("trial", i)
    test = SingulationEnv()
    while True:
        try:
            test.create_random_env(num_objects)
            break
        except:
            pass
    result = sampling(test, num_samples, num_steps)
    lst.append(result)
avg = sum(lst) / num_trials
print("avg", avg)
print("lst", lst)