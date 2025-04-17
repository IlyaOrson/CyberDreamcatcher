# Cyber Dreamcatcher

This repository implements a Graph Attention Network (GAT) (same architecture as [TacticAI](https://www.nature.com/articles/s41467-024-45965-x#Sec8)) as a network-aware reinforcement learning policy for cyber defence.
Our work extends the Cyber Operations Research Gym ([CybORG](https://github.com/alan-turing-institute/CybORG_plus_plus)) to represent network states as directed graphs with low-level features to explore more realistic autonomous defence strategies.

<div align="center">
  <img src="https://github.com/user-attachments/assets/2a77929c-ffb1-41ab-954b-7bb024bce8c7" width="70%">
</div>

## Overview

#### Core Features
- **Topology-Aware Defence**: Processes the complete network graph structure instead of simplified flat state observations
- **Runtime Adaptability**: Handles dynamic changes in network topology as new connections appear
- **Cross-Network Generalisation**: Trained policies can be deployed to networks of different sizes
- **Enhanced Interpretability**: Defence actions can be explained through tangible network properties

#### What is included?
- Custom CybORG environment with graph-based network state representation
- GAT architecture modified for compatibility with policy gradient methods
- Empirical evaluation for assessing policy generalisation vs. specialised training across varying network sizes

> [!NOTE]
> This is a research project that serves as a proof-of-concept towards more realistic network environments in cyber defence.
> Our implementation uses the low-level structure of the CybORG v2.1 simulator as a practical context, but the technique itself can be adapted to other simulators with comparable complexity.

## Setup

<details>
  <summary><i>Expand</i></summary>

We used and recommend [pixi](https://github.com/prefix-dev/pixi) to setup a reproducible project with predefined tasks.

Clone this repo recursively to clone the CybORG v2.1 simulator and CAGE 2 reference submissions as submodules.

```bash
git clone https://github.com/IlyaOrson/CyberDreamcatcher.git --recurse-submodules -j4
```

Install the dependencies of the project in a local environment.

```bash
cd CyberDreamcatcher
pixi install  # setup from pixi.toml file
```

Then install the submodules as local packages avoiding using pip to deal with dependencies.

```bash
# install environments from git submodules as a local packages
pixi run install-cyborg  # CybORG 2.1 + update to gymnasium API

# OR a debugged version from The Alan Turing Institute (https://github.com/alan-turing-institute/CybORG_plus_plus)
pixi run install-cyborg-debugged

# install troublesome dependencies without using pip to track their requirements
pixi run install-sb3  # stable baselines 3
```

Voila! An activated shell within this environment will have all dependencies working together.

```bash
pixi shell  # activate shell
python -m cyberdreamcatcher  # try out a single environment simulation
```

> [!TIP]
> If you would like to use other project management tool, the list of dependencies and installation tasks are available in [pixi.toml](pixi.toml).
> Untested environment files are provided for uv/pip ([pyproject.toml](pyproject.toml)) and for conda/mamba ([conda_env.yml](conda_env.yml)).
> Make sure to manually ignore the deps set by CybORG/SB3 when installing it locally.


</details>

## Functionality

We include predefined tasks that can be run to make sure everything is working:

```bash
pixi task list  # displays available tasks

pixi run test-cyborg  # run gymnasium-based cyborg tests

pixi run eval-cardiff  # CAGE 2 winner policy inference (simplified and flattened observation space)
```

> [!TIP]
> [Hydra](https://hydra.cc/docs/tutorials/basic/running_your_app/working_directory/) is used to handle the inputs and outputs of every script.
> The available parameters for each task are accessible with the `--help` flag.
> The content generated per execution is stored in the `outputs/` directory with subdirectories per timestamp of execution.
> The hyperparameters used in each run are registered in a hidden subfolder `.hydra/` within the generated output folder.
> Tensorboard is used to track interesting metrics, just specify the correct hydra output folder as the logdir: `tensorboard --logdir=outputs/...`

### Graph Layout

Quickly visualise the graph layout setup in the cage 2 challenge scenario file,
and the graph observations received by a random GAT policy.

```bash
pixi run plot-network scenario=Scenario2  # see --help for hyperparameters
```

> [!WARNING]
> This is the layout we expect from the simulator configuration and the actions available to the meander agent, but CybORG does not enforce these connection layout at runtime.
> Connections between other subnets to User0 appear sporadically (unexpected), possibly as a hackish way of flagging the interaction of the meander agent with deployed decoys.

### Training

<!-- #### PPO  (see issue [#20](https://github.com/IlyaOrson/CyberDreamcatcher/issues/20))

We adapted the CleanRL implementation of PPO to handle our graph observation space, which is not compatible with gymnasium restrictions.

```bash
pixi run train-gnn-ppo  # see --help for hyperparameters
``` -->

<!-- #### REINFORCE -->

We include an implementation of the REINFORCE algorithm with a normalised rewards-to-go baseline.
This is a bit slow since it samples a lot of episodes with a fixed policy to estimate the gradient before taking an optimisation step.

```bash
pixi run train-gnn-reinforce  # see --help for hyperparameters
```

#### A high-level flattened observation space reference

This script trains a MLP policy with PPO using Stable Baselines 3.
The observation space used is the original CAGE 2 observation space - a flattened high-level representation of the network.

```bash
pixi run train-flat-sb3-ppo  # see --help for hyperparameters
```

> [!IMPORTANT]
> This SB3 MLP serves as a reference for performance, but cannot extrapolate to different network dimensions.
> A major caveat for a performance comparison with this or the CAGE 2 submissions is that the observation spaces are fundamentally different: the flattened version is a higher level representation designed for the CAGE 2 Challenge, whereas our custom graph observation uses low-level information from the CybORG simulator.
> See below for a performance comparison with CAGE 2 Challenge submissions.

### Performance

It is possible (❗) to extrapolate the performance of a trained GAT policy under different network layouts.

#### Visualise reward to go at each timestep

<details>
  <summary><i>Expand</i></summary>

Specify a scenario to sample episodes from and optionally the weights of a pretrained policy (potentially trained on a different scenario).

```bash
# The default behaviour is to use a random policy on "Scenario2".
pixi run plot-performance

# This will compare the performance of a trained policy
# with a random policy on the scenario used for training
pixi run plot-performance policy_weights="path/to/trained_params.pt"
```

<div align="center">
  <img src="https://github.com/user-attachments/assets/9c9f0351-25cc-4eb9-98fe-2b1350c5a56a" width="70%">
</div>

</details>

#### Generalisation to different networks

<details>
  <summary><i>Expand</i></summary>

The objective is to compare the optimality gap trade-off between the extrapolation of a policy against a policy trained from scratch in each scenario.
Specify the path to the trained policy to be tested and array of paths of the specialised policies to compare it to; the corresponding scenarios are loaded from the logged configuration.

```bash
# add --help to see the available options
pixi run plot-generalisation policy_weights=path/to/trained_params.pt local_policies=[path/to/0/trained_params.pt,path/to/1/trained_params.pt,path/to/3/trained_params.pt, ...]
```

<div align="center">
  <img src="https://github.com/user-attachments/assets/cce8ca1a-7061-4b1b-9a93-cfae0c7dec8a" width="100%">
</div>

</details>

#### Comparison to CAGE 2 submissions

<details>
  <summary><i>Expand</i></summary>

For a detailed description of the CAGE 2 Challenge, see [this preprint](https://arxiv.org/abs/2309.07388).

For a complete list of CAGE 2 submission standings, see [here](https://github.com/cage-challenge/cage-challenge-2).

|   Scenario 2<br>Red Agent: Meander<br>Steps: 30   | Penalty<br>mean |  Observation Space 	| Structural Generalization |
|:------------------------------------------------:	|:---------------:|:------------------:	|:-------------------------:|
|                   CAGE2 Winner                   	|       ~ 6       |   High-level Flat 	|             No            |
|          Stable Baselines 3<br>MLP + PPO         	|       ~ 12      |   High-level Flat 	|             No            |
|        **CyberDreamcatcher**<br>REINFORCE        	|       ~ 18      |   Low-level Graph 	|         Reasonable        |
|                 CAGE2 CSS Random                 	|       ~ 33      |   High-level Flat 	|             N/A           |
|                 CAGE2 CSS Sleeper                	|       ~ 39      |   High-level Flat 	|             N/A           |

</details>
