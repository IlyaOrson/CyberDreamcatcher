from dataclasses import dataclass

import hydra
from hydra.core.config_store import ConfigStore
# import matplotlib.pyplot as plt

from cyberdreamcatcher.env import GraphWrapper
from cyberdreamcatcher.plots import plot_feasible_connections


@dataclass
class Config:
    scenario: str = "Scenario2"


cs = ConfigStore.instance()
cs.store(name="config", node=Config)


@hydra.main(version_base=None, config_name="config")
def main(cfg: Config):
    env = GraphWrapper(scenario=cfg.scenario, render_mode=None)
    # with plt.xkcd():
    plot_feasible_connections(env, show=True, block=True)
    plot_feasible_connections(env, filepath="./outputs/layout.pdf")


main()
