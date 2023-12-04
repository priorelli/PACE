# Flexible intentions

<p align="center">
  <img src="/reference/images/env.png">
</p>

This is the project related to the papers [Flexible Intentions: An Active Inference Theory](https://www.frontiersin.org/articles/10.3389/fncom.2023.1128694/full) and [Intention Modulation for Multi-step Tasks in Continuous Time Active Inference](https://link.springer.com/chapter/10.1007/978-3-031-28719-0_19). It contains a proposal about encoding environmental entities (e.g., a target to reach or a previously memorized home button) and realizing dynamic goal-directed behavior such as object tracking, and some analyses on multisensory integration for movements.

## HowTo

# Start simulation

The simulation can be launched through **main.py** either with the option "-m" for manual control, "-s" for the active inference agent with default parameters, or "-a" for choosing the parameters from the console. If no option is specified, the default simulation will be launched. For the manual control simulation, the arm can be moved with the keys Z, X, LEFT, RIGHT, UP and DOWN.

The dataset for the VAE is generated through the option "-g", while "-t" will run a benchmark test on the trained VAE.

Plots can be generated through **plot.py**, either with the option "-a" for the free energy derivatives, "-d" for the belief trajectories, "-f" for the final positions of the hand, "-g" for the VAE gradients, "-p" for angles and velocities, "-s" for the scores, or "-v" for generating a video of the simulation.

# Advanced configuration

More advanced parameters can be manually set from **config.py**. Both the target positions and the home button are stored in joint angle coordinates, the former in the list `targets` and the latter by the variable `home`.

The variable `task` affects the generation of target positions, and can assume the following values:
1. `test`: generates random target positions at each trial - see Figure 7;
2. `all`: generates the simulation used for Figure 6, i.e., one of the 9 fixed target positions of Figure 3A, followed by the home button position. The variable `n_reps` denotes the number of repetition per target position;
3. `single`: fixes the target to one of the 9 positions, for all trials. This position can be set with the function `set_trajectory` in **simulation/inference.py**.


## Required Libraries

matplotlib==3.6.2

numpy==1.24.1

Pillow==9.4.0

pyglet==1.5.23

seaborn==0.12.2

torch==1.12.0.dev20220205+cu113
