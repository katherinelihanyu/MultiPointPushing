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
for i in range(num_steps):
    best_result, best_action = test.best_sequential_sample(num_trials, com_only, max_step=3, metric="count threshold",
                                                           sum_path=path)
    test.step(best_action[0], best_action[1],"./data/sequential",display=True)
print("result", best_result)
print("actions", best_action)
