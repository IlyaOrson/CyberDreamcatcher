# Cyber Dreamcatcher

Graph Attention Networks (GATs) as network-aware reinforcement learning policies for cyber-defence!

![Logo](https://github.com/user-attachments/assets/73b01258-609d-4d07-b369-df323f360177)

> [!NOTE]
> This is a research project that serves as a proof-of-concept towards realistic network environments in cyber defence.
> Our implementation is based on the low-level structure of the CybORG v2.1 simulator, which is unfortunately very gimmicky.
> Our technique is applicable to other simulators of similar complexity.

## Setup

We use [pixi](https://github.com/prefix-dev/pixi) to setup a reproducible environment with predefined tasks.
If you would like to use other project management tool, the list of dependencies and tasks are available in [pixi.toml](pixi.toml).

Clone this repo recursevely to clone the custom CybORG v2.1 environment and Cage 2 reference submissions as submodules.

```bash
git clone https://github.com/IlyaOrson/CyberDreamcatcher.git --recurse-submodules -j4
```

We need to install the dependencies of the project in a local environment.

```bash
cd CyberDreamcatcher
pixi install  # setup from pixi.toml file
```

Then install the submodules as local packages avoiding using pip to deal with dependencies.

```bash
# install environments from git submodule as a local packages
pixi run install-cyborg  # CybORG 2.1 + update to gymnasium API

# install troublesome dependencies without using pip to track their requirements
pixi run install-sb3  # stable baselines 3
```

Voila! An activated shell within this environment will have all dependencies working together.

```bash
pixi shell  # activate shell
python -m cyberdreamcatcher  # try out environment simulation
```

## Functionality

We include predefined tasks that can be run to make sure everything is working:

```bash
pixi task list  # displays available tasks

pixi run test-cyborg  # run gymnasium-based cyborg tests

pixi run eval-cardiff  # cage 2 winner policy inference (simplified and flattened observation space)
```

> [!TIP]
> [Hydra](https://hydra.cc/docs/tutorials/basic/running_your_app/working_directory/) is used to handle the inputs and outputs of every script.
> All the available parameters for each task are accessible with the `--help` flag.
> The content generated per execution is stored in the `outputs/` directory with subdirectories per timestamp of execution.
> The hyperparameters used in each run are registered in a hidden subfolder `.hydra/` within the generated output folder.
> Tensorboard is used to track interesting metrics, just specify the correct hydra output folder as the logdir: `tensorboard --logdir=outputs/...`

### Graph layout

Quickly visualize the graph layout setup in the cage 2 challenge scenario file,
and the graph observations received by a random GAT policy.

```bash
pixi run plot-network scenario=Scenario2  # see --help for hyperparameters
```

> [!WARNING]
> This is the layout we expect from the simulator configuration... BUT unfortunately this is not respected by CybORG.

### Training

#### PPO

We adapted the CleanRL implementation of PPO to handle our graph observation space, which is not compatible with gymnasium restrictions.

```bash
pixi run train-gnn-ppo  # see --help for hyperparameters
```

#### REINFORCE

We also include an implementation of the REINFORCE algorithm with a normalized rewards-to-go baseline for sanity check.
This is slow since it samples a lot of episodes with a fixed policy to estimate the gradient before taking an optimization step.

```bash
pixi run train-gnn-reinforce  # see --help for hyperparameters
```

### Flat observation space + MLP + SB3-PPO

This trains the canonical flat observation space and a MLP policy we use SB3 with PPO:

```bash
pixi run train-flat-sb3-ppo  # see --help for hyperparameters
```

> [!IMPORTANT]
> A direct performance comparison is not possible because the observation space is different due to the graph inductive bias.

### Performance

It is possible (❗) to extrapolate the performance of a trained policy under different network layouts.

#### Visualize reward to go at each timestep

Specify a scenario to sample episodes from and optionally the weights of a pretrained policy (potentially trained on a different scenario).

```bash
# The default behaviour is to use a random policy on "Scenario2".
pixi run plot-performance

pixi run plot-performance policy_weights="path/to/trained_params.pt" scenario="Scenario2_+_User6"
```
![joyplot](https://github.com/user-attachments/assets/ad6b7ef0-7ebc-4d92-9281-c2c48337a01e)

#### Generalization to different networks

Specify the path to a trained policy to be loaded, as well as the name of a specific scenario in `scenarios/` to sample the performance on.

```bash
# add --help to see the available options
pixi run plot-generalization policy_path="path/to/trained_params.pt"
```

It is possible to provide the directory to other trained policies specialized per scenario, in order to compare the optimality gap trade-off between the extrapolation of a policy and a specialized policy per scenario.

```bash
# add --help to see the available options
pixi run plot-generalization policy_path="path/to/trained_params.pt specialized_policies=[/path/to/dir1,/path/to/dir2,/path/to/dir3]"
```
