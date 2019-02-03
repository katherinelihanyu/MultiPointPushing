from helper import *
from policies import *
from prune import *
from singulation_env_simulation import *
num_objects=5
path="."
test = SingulationEnv()
test.create_random_env(num_objects)
num_trials = 3
num_steps = 3
def sampling(test, num_trials, num_steps):
    for i in range(num_steps):
        best_result, best_action = test.best_sequential_sample(num_trials, com_only, max_step=3, metric="count threshold",
                                                               sum_path=path)
        test.step(best_action[0], best_action[1], "./data/sequential_sampling_%d"%i, display=True)
    return best_result, best_action

for i in range(num_steps):
    curr_pos = test.save_curr_position()
    best_pts = test.prune_best(no_prune, "count threshold", curr_pos)
    if best_pts is None:
        break
    best_summary = test.collect_data_summary(best_pts[0], best_pts[1], img_path="./data/sequential_greedy_%d"%i)
    best_dist = best_summary["count threshold after push"] - best_summary["count threshold before push"]
    print("count threshold", best_summary["count threshold after push"])
    if best_dist <= 0:
        break


