# Flexible intentions in active inference

<p align="center">
  <img src="/reference/images/env.png">
</p>

This is the project related to the paper [Flexible Intentions: An Active Inference Theory](https://www.frontiersin.org/articles/10.3389/fncom.2023.1128694/full). It contains a proposal about encoding environmental entities (e.g., a target to reach or a previously memorized home button) and realizing dynamic goal-directed behavior such as object tracking, and some analyses on multisensory integration for movements. The paper [Intention Modulation for Multi-step Tasks in Continuous Time Active Inference](https://link.springer.com/chapter/10.1007/978-3-031-28719-0_19) extends this model by introducing a fixed multi-step behavior (e.g., reaching multiple target positions in sequence).

Video simulations are found [here](https://priorelli.github.io/projects/1_flexible_intentions/).

Check [this](https://priorelli.github.io/blog/) and [this](https://priorelli.github.io/projects/) for additional guides and projects.

This study has received funding from the [MAIA project](https://site.unibo.it/maia-fetproact/en), under the European Union's Horizon 2020 Research and Innovation programme.

## HowTo

### Start the simulation

The simulation can be launched through *main.py*, either with the option `-m` for manual control, `-s` for the active inference agent with default parameters, or `-a` for choosing the parameters from the console. If no option is specified, the last one will be launched. For the manual control simulation, the arm can be moved with the keys `Z`, `X`, `LEFT`, `RIGHT`, `UP` and `DOWN`.

The dataset for the VAE is generated through the option `-g`, while `-t` will run a benchmark test on the trained VAE. For each datapoint, a random target size is sampled from the variable `target_min_max` in *config.py*.

Plots can be generated through *plot.py*, either with the option `-a` for the free energy derivatives, `-d` for the belief trajectories, `-f` for the final positions of the hand, `-g` for the VAE gradients, `-p` for angles and velocities, `-s` for the scores, or `-v` for generating a video of the simulation.

The folder *reference/video/* contains a few videos about target tracking, movements with or without visual input, and dynamic onset policy.

### Advanced configuration

More advanced parameters can be manually set from *config.py*. Custom log names are set with the variable `log_name`. The number of trials and steps can be set with the variables `n_trials` and `n_steps`, respectively.

Both the target positions and the home button are stored in joint angle coordinates, the former in the list `targets` and the latter by the variable `home`.

The variable `task` affects the generation of target positions, and can assume the following values:
1. `test`: generates random target positions at each trial - see Figure 7;
2. `all`: generates the simulation used for Figure 6, i.e., one of the 9 fixed target positions of Figure 3A, followed by the home button position. The variable `n_reps` denotes the number of repetition per target position;
3. `single`: fixes the target to one of the 9 positions, for all trials. This position can be set with the function `set_trajectory` in *simulation/inference.py*.

The variable `context` specifies whether (`dynamic`) or not (`static`) the target moves. The target velocity is set by `target_vel`.

The variable `phases` chooses the movement onset policy of the agent (`immediate`, `fixed`, or `dynamic`) as defined in the paper.

The arm configuration is defined through the dictionary `joints`. The value `link` specifies the joint to which the new one is attached; `angle` encodes the starting value of the joint; `limit` defines the min and max angle limits.

### Agent

The active inference simulation involves the scripts *simulation/inference.py* and *simulation/agent.py*. The former contains a subclass of `Window` in *environment/window.py*, which is in turn a subclass `pyglet.window.Window`. The only overriden function is `update`, which defines the instructions to run in a single cycle. Specifically, the subclass `Inference` initializes the agent and the sequence of target positions; during each update, it retrieves proprioceptive and visual observations through functions defined in *environment/window.py*, calls the function `inference_step` of *simulation/agent.py*, and finally moves the arm and the target.

The function `inference_step` of the class `Agent` in *simulation/agent.py* contains all the instructions of Algorithm 1. In particular, the function `get_p` returns a visual prediction through the VAE, and a proprioceptive prediction through the matrix `G_p`. Note that the input and output of the VAE is stored in an additional list, which is needed for the successive error backpropagation. Note also this function returns a velocity prediction `p_vel`, which however is not used in the analyses of the paper. The function `get_h` returns future beliefs computed through the intention matrices stored in `I` (encoding `I_t` and `I_h` of the paper). The list `I` can thus be used to realize custom dynamic behaviors. Functions `get_e_s` and `get_e_mu` compute sensory and dynamics prediction errors, respectively. The function `get_likelihood` backpropagates the sensory errors toward the belief, multiplying them by the precisions encoded in the list `pi_s` and the variable `alpha`. Finally, the function `mu_dot` computes the total belief update, also considering the backward and forward errors of the dynamics function.

Useful trajectories computed during the simulations are stored through the class `Log` in *environment/log.py*.

Note that all the variables are normalized between 0 and 1 to ensure that every contribution to the belief updates has the same magnitude.

## Required libraries

matplotlib==3.6.2

numpy==1.24.1

Pillow==9.4.0

pyglet==1.5.23

seaborn==0.12.2

torch==1.12.0.dev20220205+cu113
