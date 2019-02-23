import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random

from helper import *
from policies import *
from prune import *
from singulation_env_simulation import *

def sampling_every_step(summary, data_path, num_samples, num_steps, reuse, reuse_path=None, metric="count soft threshold", timeit=False):
    first_time = datetime.datetime.now().replace(microsecond=0)
    cur_time = datetime.datetime.now().replace(microsecond=0)
    reuse_step_path = None
    env = SingulationEnv()
    env.load_env(summary)
    for i in range(num_steps):
        print("step", i)
        actual_step_path = os.path.join(data_path, "actual_step"+str(i))
        if not os.path.exists(actual_step_path):
            os.makedirs(actual_step_path)
        if reuse:
            reuse_step_path = os.path.join(reuse_path, "actual_step"+str(i))
            if not os.path.exists(reuse_step_path):
                os.makedirs(reuse_step_path)
        best_result, best_pts, best_before, best_after_step0 = env.best_sequential_sample(num_samples, no_prune, reuse=reuse, max_step=num_steps-i,
                                                           data_path=actual_step_path, reuse_path=reuse_step_path, metric=metric)
        print("sampled")
        print(metric, "before push", best_before)
        print(metric, "after push", best_after_step0)
        print(metric, "total reward", best_result)
        if reuse:
            best_summary = env.collect_data_summary(best_pts[0], best_pts[1],
                                                    img_path=os.path.join(reuse_step_path, "render"), display=False,
                                                    sum_path=reuse_step_path)
        else:
            best_summary = env.collect_data_summary(best_pts[0], best_pts[1], img_path=os.path.join(actual_step_path, "render"), display=True, sum_path=actual_step_path)
        print("actual")
        print(metric + " before push", best_summary[metric + " before push"])
        print(metric + " after push", best_summary[metric + " after push"])
        if timeit:
            print("sampling step %d took"%i, datetime.datetime.now().replace(microsecond=0) - cur_time)
            cur_time = datetime.datetime.now().replace(microsecond=0)
    if timeit:
        print("sampling took", datetime.datetime.now().replace(microsecond=0) - first_time)
    return best_summary[metric + " after push"]

def sampling_open_loop(summary, data_path, num_samples, num_steps, reuse, metric="count soft threshold", timeit=False):
    first_time = datetime.datetime.now().replace(microsecond=0)
    cur_time = datetime.datetime.now().replace(microsecond=0)
    env = SingulationEnv()
    env.load_env(summary)
    actual_step_path = os.path.join(data_path, "samples")
    if not os.path.exists(actual_step_path):
        os.makedirs(actual_step_path)
    best_result, actions, best_before, best_after_step0  = env.best_sequential_sample(num_samples, no_prune, reuse=reuse, max_step=num_steps,
                                                   data_path=actual_step_path, metric=metric, open_loop=True)
    print("sampled")
    print(metric, "before push", best_before)
    print(metric, "after push", best_after_step0)
    print(metric, "total reward", best_result)
    for i in range(len(actions)):
        best_pts = actions[i]
        print("step", i)
        actual_step_path = os.path.join(data_path, "actual_step" + str(i))
        if not os.path.exists(actual_step_path):
            os.makedirs(actual_step_path)
        print("best_pts[0]", best_pts[0])
        print("best_pts[1]", best_pts[1])
        best_summary = env.collect_data_summary(best_pts[0], best_pts[1], img_path=actual_step_path, display=False, sum_path=actual_step_path)
        print("actual")
        print(metric + " before push", best_summary[metric + " before push"])
        print(metric + " after push", best_summary[metric + " after push"])
        if timeit:
            print("sampling step %d took"%i, datetime.datetime.now().replace(microsecond=0) - cur_time)
            cur_time = datetime.datetime.now().replace(microsecond=0)
    if timeit:
        print("sampling took", datetime.datetime.now().replace(microsecond=0) - first_time)
    return best_summary[metric + " after push"]

# Sample num_samples num_steps pushes on a specific environment
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

def greedy_sequential(summary, num_steps, img_path, metric="count soft threshold", save_img = True):
    cur_time = time.time()
    test = SingulationEnv()
    test.load_env(summary)
    for i in range(num_steps):
        print("step", i)
        actual_step_path = os.path.join(img_path, "step"+str(i))
        if not os.path.exists(actual_step_path):
            os.makedirs(actual_step_path)
        if not os.path.exists(os.path.join(actual_step_path, "render")):
            os.makedirs(os.path.join(actual_step_path, "render"))
        if not os.path.exists(os.path.join(actual_step_path, "summary")):
            os.makedirs(os.path.join(actual_step_path, "summary"))
        curr_pos = test.save_curr_position()
        best_pts = test.prune_best(prune_method=no_prune, metric=metric, position=curr_pos)
        if best_pts is None:
            print("best_pts is None")
            break
        if save_img:
            best_summary = test.collect_data_summary(
                best_pts[0], best_pts[1], img_path=os.path.join(actual_step_path, "render"), display=True,
                sum_path=os.path.join(actual_step_path, "summary"))
        else:
            best_summary = test.collect_data_summary(best_pts[0], best_pts[1])
        best_dist = best_summary[metric + " after push"] - best_summary[metric + " before push"]
        print("actual",)
        print(metric + " before push", best_summary[metric + " before push"])
        print(metric + " after push", best_summary[metric + " after push"])
        print("greedy step %d took"%i, time.time()-cur_time)
        cur_time = time.time()
        if best_dist <= 0:
            print("best_dist <= 0")
            return best_summary[metric + " before push"]
    return best_summary[metric + " after push"]

def run_experiments(num_trials,data_path, func, reuse):
    lst = []
    for i in range(num_trials):
        print("heap", i)
        path_i = os.path.join(data_path, "env" + str(i))
        with open(os.path.join(path_i, "env.json"), "r") as read_file:
            summary = json.load(read_file)
        result = func(summary, path_i, reuse)
        lst.append(result)
    return lst

def plot(num_samples_lst, means, stds, num_objects,num_trials,num_steps,path):
    plt.figure()
    plt.errorbar(num_samples_lst, means, yerr=stds)
    plt.title("%d-push sampling with %d objects and %d trials"%(num_steps,num_objects,num_trials))
    plt.xlabel('Number of samples')
    plt.ylabel('Count threshold')
    plt.savefig(path)


def generate_roll_outs(num_trials, num_roll_outs, data_path):
    for i in range(num_trials):
        before1 = []
        after1 = []
        before2 = []
        after2 = []
        before3 = []
        after3 = []
        path_i = os.path.join(data_path, "env" + str(i))
        with open(os.path.join(path_i, "env.json"), "r") as read_file:
            summary = json.load(read_file)
        env = SingulationEnv()
        env.load_env(summary)
        pt_lst1 = no_prune(env)
        action1 = random.choice(pt_lst1)
        best_summary = env.collect_data_summary(action1[0], action1[1])
        bf = best_summary[metric + " before push"]
        before1.append(bf)
        # print("step0", metric + " before push", bf)
        af = best_summary[metric + " after push"]
        after1.append(af)
        # print("step0", metric + " after push", af)
        pt_lst2 = no_prune(env)
        action2 = random.choice(pt_lst2)
        best_summary = env.collect_data_summary(action2[0], action2[1])
        bf = best_summary[metric + " before push"]
        before2.append(bf)
        # print("step1", metric + " before push", bf)
        af = best_summary[metric + " after push"]
        after2.append(af)
        # print("step1", metric + " after push", af)
        pt_lst3 = no_prune(env)
        action3 = random.choice(pt_lst3)
        best_summary = env.collect_data_summary(action3[0], action3[1])
        bf = best_summary[metric + " before push"]
        before3.append(bf)
        # print("step2", metric + " before push", bf)
        af = best_summary[metric + " after push"]
        after3.append(af)
        # print("step2", metric + " after push", af)
        for i in range(num_roll_outs):
            env = SingulationEnv()
            env.load_env(summary)
            best_summary = env.collect_data_summary(action1[0], action1[1])
            bf = best_summary[metric + " before push"]
            before1.append(bf)
            # print("step0", metric + " before push", bf)
            af = best_summary[metric + " after push"]
            after1.append(af)
            # print("step0", metric + " after push", af)
            best_summary = env.collect_data_summary(action2[0], action2[1])
            bf = best_summary[metric + " before push"]
            before2.append(bf)
            # print("step1", metric + " before push", bf)
            af = best_summary[metric + " after push"]
            after2.append(af)
            # print("step1", metric + " after push", af)
            best_summary = env.collect_data_summary(action3[0], action3[1])
            bf = best_summary[metric + " before push"]
            before3.append(bf)
            # print("step2", metric + " before push", bf)
            af = best_summary[metric + " after push"]
            after3.append(af)
            # print("step2", metric + " after push", af)
        if not after1 == before2:
            print(i, "not after1 == before2")
        if not after2 == before3:
            print(i, "not after2 == before3")
        print([np.std(x) for x in [before1, after1, before2, after2, before3, after3]])

num_objects = 10
num_trials = 3
num_steps = 1
beg_time = datetime.datetime.now().replace(microsecond=0)
num_samples = 1000
data_path= "/nfs/diskstation/katherineli/sampling_open1"
metric = "count soft threshold"
num_roll_outs = 10
num_trials = 100

returns = run_experiments(num_trials, data_path="/nfs/diskstation/katherineli/sampling_open1", reuse=False,
                          func=lambda summary, data_path, reuse: sampling_open_loop(summary=summary, data_path=data_path,
                                                                                        num_samples=num_samples,
                                                                                        num_steps=num_steps,
                                                                                        reuse=reuse,
                                                                                        metric="count soft threshold",
                                                                                        timeit=True))
m = np.mean(returns)
s = np.std(returns)
print("%d samples, mean: %.2f, std: %.2f"%(num_samples, m, s))
print("returns", returns)
time_elapsed = datetime.datetime.now().replace(microsecond=0) - beg_time
print("Time elapsed:", time_elapsed)
with open("/nfs/diskstation/katherineli/sampling_open1/2400_samples_time1.pickle", 'wb') as f:
    pickle.dump(time_elapsed, f)
with open("/nfs/diskstation/katherineli/sampling_open1/2400_samples_returns1.pickle", 'wb') as f:
    pickle.dump(returns, f)


# returns = run_experiments(num_trials, data_path="/nfs/diskstation/katherineli/greedy", reuse=True,
#                           func=lambda summary, data_path, reuse: greedy_sequential(summary=summary, num_steps=num_steps,
#                                                                                         img_path = data_path,
#                                                                                         metric="count soft threshold"))
# m = np.mean(returns)
# s = np.std(returns)
# print("returns", returns)
# print("Greedy: mean: %.2f, std: %.2f"%(m, s))
# print("Time elapsed:", datetime.datetime.now().replace(microsecond=0) - beg_time)
# with open("/nfs/diskstation/katherineli/greedy/returns1.pickle", 'wb') as f:
#     pickle.dump(returns, f)
#
# returns = run_experiments(num_trials, data_path="/nfs/diskstation/katherineli/sampling1", reuse=False,
#                           func=lambda summary, data_path, reuse: sampling_every_step(summary=summary, data_path=data_path,
#                                                                                         num_samples=num_samples,
#                                                                                         num_steps=num_steps,
#                                                                                         reuse=reuse,
#                                                                                         metric="count soft threshold",
#                                                                                         timeit=True))
# m = np.mean(returns)
# s = np.std(returns)
# print("%d samples, mean: %.2f, std: %.2f"%(num_samples, m, s))
# print("returns", returns)
# time_elapsed = datetime.datetime.now().replace(microsecond=0) - beg_time
# print("Time elapsed:", time_elapsed)
# with open("/nfs/diskstation/katherineli/sampling/2400_samples_time1.pickle", 'wb') as f:
#     pickle.dump(time_elapsed, f)
# with open("/nfs/diskstation/katherineli/sampling/2400_samples_returns1.pickle", 'wb') as f:
#     pickle.dump(returns, f)

# returns = run_experiments(num_trials, data_path="/nfs/diskstation/katherineli/sampling_greedy", reuse=True,
#                           func=lambda summary, data_path, reuse: sampling_last_step_greedy(summary=summary, data_path=data_path,
#                                                                                         num_samples=num_samples,
#                                                                                         num_steps=num_steps,
#                                                                                         reuse=reuse,
#                                                                                         metric="count soft threshold",
#                                                                                         timeit=True))
# m = np.mean(returns)
# s = np.std(returns)
# print("%d samples, mean: %.2f, std: %.2f"%(num_samples, m, s))
# print("Time elapsed:", datetime.datetime.now().replace(microsecond=0) - beg_time)
# with open("/nfs/diskstation/katherineli/sampling_greedy/1275_samples.pickle", 'wb') as f:
#     pickle.dump(returns, f)
# mypath = "/nfs/diskstation/katherineli/sampling_open"
# returns = run_experiments(num_trials, data_path=mypath, reuse=False,
#                           func=lambda summary, data_path, reuse: sampling_open_loop(summary=summary, data_path=data_path,
#                                                                                         num_samples=num_samples,
#                                                                                         num_steps=num_steps,
#                                                                                         reuse=reuse,
#                                                                                         metric="count soft threshold",
#                                                                                         timeit=True))
# m = np.mean(returns)
# s = np.std(returns)
# print("%d samples, mean: %.2f, std: %.2f"%(num_samples, m, s))
# time_elapsed = datetime.datetime.now().replace(microsecond=0) - beg_time
# print("Time elapsed:", time_elapsed)
# with open(os.path.join(mypath, '2400samples_open_loop.pickle'), 'wb') as f:
#     pickle.dump(returns, f)
# with open(os.path.join(mypath, '2400samples_time_open_loop.pickle'), 'wb') as f:
#     pickle.dump(time_elapsed, f)
#
# num_samples_lst = [675, 1250, 1825]
# returns = []
# means = []
# stds = []
# curr_time = datetime.datetime.now().replace(microsecond=0)
# path = "/nfs/diskstation/katherineli/sampling"
# for num_samples in num_samples_lst:
#     print("sample size:", num_samples)
#     returns = run_experiments(num_trials, data_path=path, reuse=True,
#                               func=lambda summary, data_path, reuse: sampling_every_step(summary=summary, data_path=data_path,
#                                                                                          reuse_path=os.path.join(path, "%dsamples"%num_samples),
#                                                                                             num_samples=num_samples,
#                                                                                             num_steps=num_steps,
#                                                                                             reuse=reuse,
#                                                                                             metric="count soft threshold",
#                                                                                             timeit=True))
#     m = np.mean(returns)
#     s = np.std(returns)
#     print("%d samples, mean: %.2f, std: %.2f"%(num_samples, m, s))
#     time_elapsed = datetime.datetime.now().replace(microsecond=0) - curr_time
#     print("Time elapsed:", time_elapsed)
#     means.append(m)
#     stds.append(s)
#     with open("/nfs/diskstation/katherineli/sampling/%d_samples_time.pickle"%num_samples, 'wb') as f:
#         pickle.dump(time_elapsed, f)
#     with open("/nfs/diskstation/katherineli/sampling/%d_samples_returns.pickle"%num_samples, 'wb') as f:
#         pickle.dump(returns, f)
#     curr_time = datetime.datetime.now().replace(microsecond=0)
#
# print("means", means)
# print("stds", stds)
# print("Time elapsed:", datetime.datetime.now().replace(microsecond=0) - beg_time)
# with open('/nfs/diskstation/katherineli/sampling/sampling_result.pickle', 'wb') as f:
#     pickle.dump([means,stds], f)
# means = [5.52,6.42,6.75,6.8,6.9,7.19, 6.95, 6.93, 7.14]
# stds = [0.97,1.04,0.79,0.81,1.1,1.2,0.99,0.93, 0.97]
