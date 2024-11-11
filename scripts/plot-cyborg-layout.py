from dataclasses import dataclass

import hydra
from hydra.core.config_store import ConfigStore
# import matplotlib.pyplot as plt

from cyberdreamcatcher.env import GraphWrapper
from cyberdreamcatcher.plots import plot_feasible_connections


@dataclass
class Cfg:
    scenario: str = "Scenario2"

# Registering the Config class with the expected name 'args'.
# https://hydra.cc/docs/tutorials/structured_config/minimal_example/
cs = ConfigStore.instance()
cs.store(name="args", node=Cfg)

@hydra.main(version_base=None, config_name="args", config_path=None)
def main(cfg: Cfg):
    env = GraphWrapper(scenario=cfg.scenario, render_mode=None)
    # with plt.xkcd():
    plot_feasible_connections(env, show=True, block=True)
    plot_feasible_connections(env, filepath="./outputs/layout.pdf")


main()
