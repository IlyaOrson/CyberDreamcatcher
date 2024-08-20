# Cyber Dreamcatcher

GNNs as network-aware policies for cyber-defence in RL environments!

<p align="center">
  <img width="460" src="https://github.com/user-attachments/assets/73b01258-609d-4d07-b369-df323f360177">
</p>

## Setup

Clone this repo recursevely to clone the custom CybORG v2.1 environment and Cage 2 reference submissions as submodules.

```bash
git clone https://github.com/IlyaOrson/CyberDreamcatcher.git --recurse-submodules -j3
```

We use [pixi](https://github.com/prefix-dev/pixi) to setup a reproducible environment.
First we need to install the dependencies of the project in a local environment.

```bash
cd CyberDreamcatcher
pixi install  # setup from pixi.toml file
```

Afterwards, install the submodules as local packages avoiding using pip to deal with dependencies.

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

We include predefined tasks that can be run to make sure everything is working:

```bash
pixi task list  # displays available tasks

pixi run test-cyborg  # run gymnasium-based cyborg tests

pixi run eval-cardiff  # cage 2 winner policy inference (simplified and flattened observation space)
```

## Graph layout

Quickly visualize the graph layout setup in the cage 2 challenge scenario file,
and the graph observations received by a random GAT policy.

```bash
pixi run plot-network scenario=Scenario2
```

> [!IMPORTANT]
> This is the layout we expect to observe from the configuration... BUT this is not strictly enforced in CybORG v2.1!

## Training

### PPO

We adapted the CleanRL implementation of PPO to handle our graph observation space, which is not compatible with gymnasium restrictions.

```bash
pixi run train-gnn-ppo  # see --help for hyperparameters
```

### REINFORCE

We also include an implementation of the REINFORCE algorithm with a normalized rewards-to-go baseline for sanity check.
This is slow since it samples a lot of episodes with a fixed policy to estimate the gradient before taking an optimization step.

```bash
pixi run train-gnn-reinforce  # see --help for hyperparameters
```

### Flat observation space + MLP + SB3-PPO

This trains the canonical flat observation space and a MLP policy we use SB3 with PPO:

```bash
pixi run train-flat-sb3-ppo
```

> [!NOTE]
> A direct performance comparison is not possible because the observation space is different due to the graph inductive bias.
