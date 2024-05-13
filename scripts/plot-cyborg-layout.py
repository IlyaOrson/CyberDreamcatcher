from dataclasses import dataclass

import hydra
from hydra.core.config_store import ConfigStore

from blueskynet.env import GraphWrapper
from blueskynet.plots import plot_feasible_connections

@dataclass
class Config:
    scenario: str = "Scenario2"

cs = ConfigStore.instance()
cs.store(name="config", node=Config)

@hydra.main(version_base=None, config_name="config")
def main(cfg: Config):

    env = GraphWrapper(scenario=cfg.scenario, render_mode=None)
    plot_feasible_connections(env, show=True, block=True)

main()
