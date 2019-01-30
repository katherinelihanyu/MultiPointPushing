from helper import *
from policies import *
from prune import *
from singulation_env_simulation import *
num_objects=5
path="/Users/katherineli/Desktop/MultiPointPushing"
test = SingulationEnv()
test.create_random_env(num_objects)
data_sum = test.sequential_prune_planning_recursive(prune_method=com_only, max_step=3, metric="count threshold", sum_path=path)
print(data_sum)
