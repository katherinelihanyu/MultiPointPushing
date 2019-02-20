import matplotlib.pyplot as plt
import os
import json
from singulation_env_simulation import *

values = []
for x in os.listdir('/nfs/diskstation/katherineli/initial_states'):
	if os.path.isdir(x) and x.startswith('env'):
		with open(os.path.join(x, 'env.json'), 'r') as f:
			summary = json.load(f)
			env = SingulationEnv()
			env.load_env(summary)
			s = env.count_soft_threshold()
			summary["count soft threshold"] = s
			values.append(f)
		os.remove(os.path.join(x, 'env.json'))
		with open(filename, 'w') as f:
			json.dump(summary, os.path.join(x, 'env.json'))

plt.hist(values)
plt.show()