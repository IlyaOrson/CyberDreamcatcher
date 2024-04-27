# GNN-cyber-defence

GNN as network-aware policies for cyber-defence RL environments.

## Setup

Clone this repo recursevely to clone the environments (CybORG, YawningTitan) as submodules simultaneously.

```bash
git clone https://github.com/IlyaOrson/GNN-cyber-defence.git --recurse-submodules -j3
```

We use [pixi](https://github.com/prefix-dev/pixi) to setup a reproducible environment.
First one needs to install the dependencies in a local environment, activate a shell within it and afterwards install the submodules as local packages as well as troublesome dependencies.

```bash
cd GNN-cyber-defence
pixi install  # setup from pixi.toml file
```

Afterwards a shell within this environment will have all dependencies available.

```bash
pixi shell
```

From the activated shell, install the local submodules and troublesome pip dependencies .

```bash
# install environments from git submodule as a local packages
pixi run install-cyborg  # CybORG 2.1 + update to gymnasium API

# install troublesome dependencies without using pip to track their requirements
pixi run install-custom-sb3  # stable baselines 3 + adaptation to GNN policies and graph environment
```

Voila!


There are other predefined tasks that can be run to make sure everything is working:

```bash
# pixi run <TAB>  # displays options

pixi run test-cyborg  # run gymnasium-based cyborg tests

pixi run eval-cardiff  # cage 2 winner policy inference
```

## Graph layout

Quickly visualize the graph layout setup in the cage 2 challenge scenario file,
and the graph observations received by a random GNN policy.

```bash
python -m blueskynet
```

## Training

Train a GNN policy with a REINFORCE algorithm with a normalized rewards-to-go baseline.
This is very slow since it samples a lot of episodes with a fixed policy to estimate the gradient before taking an optimization step.

```bash
pixi run train-gnn-reinforce  # see --help for hyperparameters
```

To get a comparison for optimality and performance with a standard flat observation space and a MLP policy we use SB3 with PPO:

```bash
pixi run train-flat-sb3-ppo
```

## Notes

To see the notes on the project locally you will need [quarto](https://quarto.org):

```bash
quarto render notes/
```
