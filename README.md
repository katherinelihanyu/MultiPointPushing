# Sequential Pushing

A Box2D simulation environment for sequential pushing with greedy, open-loop, and closed-loop policies.  
Click [here](https://docs.google.com/presentation/d/1SVCs2WHByOlnsANJ-5jjQuTtSxtw1dyAYsxmp3lK2Gs/edit?usp=sharing) for slides.

## Project Overview

### Code

The purpose of each relevant file is described below.

- `state.py`: Box2D definitions of objects and environments
- `algorithm.py`: run experiments
- `test_state.py`: test for stability of the environments
- `helpers.py`: geometry-related helper functions

### Policies

The project implements 3 planing policies: greedy, open-loop sampling, and closed-loop sampling.

- Greedy: At each step, performs the push with maximum reward.
- Open-loop sampling: Samples sample_size number of multi-step pushes. Select the sample with the maximum reward.
- Closed-loop sampling: At each step, samples sample_size number of multi-step pushes. Perform the first step of the sample with the maximum reward.

## Prerequisites

The simulator uses pybox2d for simulation. Click [here](https://github.com/pybox2d/pybox2d) for installation instructions.


## Testing

`test_state.py` ensures that pushes found by greedy, open-loop, and closed-loop policies are reproducible.

```
python test_state.py
```

## Experiment

Below are the intructions for running experiments.

### Step 1: Create Environments

Call `create_initial_envs` in `algorithms.py` to create the initial states.

```
create_initial_envs(start_index, end_index, num_objects, data_path)
```

### Step 2: Run Policies
`algorithm.py` contains greedy, open-loop, and closed-loop policies. Hyperparameters, including number of heaps and number of samples, can be modified inside the main function of `algorithm.py`.

```
python algorithm.py
```
The results will be printed in the terminal. If `display` is True, GIF of each push will be saved in `data_path`.

### Expected Results

Below are my results with **10 objects** and **50 heaps**. Your results may be slightly different due to different initial states.
Click [here](https://docs.google.com/presentation/d/1SVCs2WHByOlnsANJ-5jjQuTtSxtw1dyAYsxmp3lK2Gs/edit#slide=id.g4ee20b2f04_0_164) for details on the metric `count_soft_threshold`.
```
Greedy: Mean: 9.893, std: 0.100
Open-loop (200 samples): Mean: 8.98. Std: 0.40
Open-loop (2400 samples): Mean: 9.44. Std: 0.28
Open-loop (3400 samples): Mean: 9.48. Std: 0.28
Closed-loop (200 samples): Mean: 9.34. Std: 0.43
Closed-loop (2400 samples): Mean: 9.68. Std: 0.27
```

## Authors

* **[Zisu Dong](https://github.com/Jekyll1021)** - *Created the simulator with pybox2d.*
* **[Hanyu Li](https://github.com/katherinelihanyu)** - *Rewrote the simulator and implemented sequential pushing policies.*

## Acknowledgments

* Ashwin Balakrishna (PhD, UC Berkeley EECS)
* Michael Danielczuk (PhD, UC Berkeley EECS)
* Ken Goldberg (Professor, UC Berkeley EECS)
