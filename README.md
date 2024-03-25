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
pixi run install-submodules  # gymnasium compatible CybORG 2.1
# install troublesome dependencies without using pip to track their requirements
pixi run install-pip-deps  # stable baselines 3
```

Voila!
There are other predefined tasks that can be run to make sure everything is working:

```bash
# pixi run <TAB>  # displays options
pixi run test-cyborg  # run gymnasium-based cyborg tests
pixi run eval-cardiff  # cage 2 winner policy inference
```

## Notes

To see the notes on the project locally use `quarto render notes/`.
