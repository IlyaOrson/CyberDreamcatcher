# GNN-cyber-defence

GNN as network-aware policies for cyber-defence RL environments.

## Setup

Clone this repo recursevely to clone the environments (CybORG, YawningTitan) as submodules simultaneously.

```bash
git clone https://github.com/IlyaOrson/GNN-cyber-defence.git --recurse-submodules -j3
```

We use [pixi](https://github.com/prefix-dev/pixi) (version >= 0.15.2) to setup a reproducible environment.
First one needs to install the dependencies and afterwards install the submodules as local packages.

```bash
cd GNN-cyber-defence
pixi install  # install all dependencies defined in the project pixi.toml file
pixi run install-submodules  # uses pip to install the git submodules as local packages
```

Afterwards a shell within this environment will have all dependencies available.

```bash
pixi shell
```

# TODO mention predefined tasks
