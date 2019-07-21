# Sequential Pushing

A Box2D simulation environment for sequential pushing with greedy, open-loop, and closed-loop policies.  
Slides: https://docs.google.com/presentation/d/1SVCs2WHByOlnsANJ-5jjQuTtSxtw1dyAYsxmp3lK2Gs/edit?usp=sharing

## Prerequisites

The simulator uses pybox2d for simulation. Click [here](https://github.com/pybox2d/pybox2d) for installation instructions.


## Testing

`test_state.py` ensures that pushes found by greedy, open-loop, and closed-loop policies are reproducible.

```
python test_state.py
```

## Environment Creation

Call `create_initial_envs` in `algorithms.py` to create the initial states

```
create_initial_envs(start_index, end_index, num_objects, data_path)
```

## Experiments

`algorithm.py` contains greedy, open-loop, and closed-loop policies. Hyperparameters, including number of heaps and number of samples, can be modified inside the main function of `algorithm.py`.

```
python algorithm.py
```

## Authors

* **[Zisu Dong](https://github.com/Jekyll1021)** - *Created the simulator with pybox2d.*
* **[Hanyu Li](https://github.com/katherinelihanyu)** - *Rewrote the simulator and implemented sequential pushing policies.*

## Acknowledgments

* Ashwin Balakrishna
* Michael Danielczuk
