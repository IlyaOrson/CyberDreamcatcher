import inspect
from pathlib import Path
from bidict import bidict
import logging

import torch
from omegaconf import OmegaConf

import CybORG

LOGGER = logging.getLogger(__name__)


def get_scenario(name="Scenario2", from_cyborg=True):
    if from_cyborg:
        # scenario_path = inspect.getfile(CybORG)[:-10] + f"/Shared/Scenarios/{self.scenario}.yaml"
        cyborg_path = Path(inspect.getfile(CybORG)).resolve()
        scenario_dir = cyborg_path.parent / "Shared" / "Scenarios"
    else:
        scenario_dir = Path(__file__).resolve().parent.parent / "scenarios"

    scenario_path = scenario_dir / Path(name).with_suffix(".yaml")

    assert scenario_path.exists()
    LOGGER.info(f"Loaded scenario file from {scenario_path}")

    return scenario_path


def enumerate_bidict(iterable):
    "Form bidirectional mappings between categorical values and their enumeration."
    return bidict((val, idx) for idx, val in enumerate(iterable))


# taken from https://github.com/francois-rozet/torchist/
# https://github.com/pytorch/pytorch/issues/35674#issuecomment-1741608630
def ravel_multi_index(coords: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    r"""Converts a tensor of coordinate vectors into a tensor of flat indices.

    This is a `torch` implementation of `numpy.ravel_multi_index`.

    Args:
        coords: A tensor of coordinate vectors, (*, D).
        shape: The source shape.

    Returns:
        The raveled indices, (*,).
    """

    shape = coords.new_tensor(shape + (1,))
    coefs = shape[1:].flipud().cumprod(dim=0).flipud()

    return (coords * coefs).sum(dim=-1)


def available_scenarios():
    # expects the scenarios to be defined at the top level of the project
    module_dir = Path(__file__).parent.absolute()
    project_dir = module_dir.parent
    scenarios_dir = project_dir / "scenarios"
    scenarios = [file.name for file in scenarios_dir.iterdir() if file.is_file()]
    return scenarios


def load_trained_weights(policy_weights_path, trained_scenario=None):
    """Load trained weights safely and extract scenario
    from logged training file if not specified"""

    policy_path = Path(policy_weights_path)
    assert policy_path.is_file()

    policy_weights = torch.load(policy_path, weights_only=True)

    if trained_scenario:
        assert (
            trained_scenario in available_scenarios()
        ), "Provided scenario is not predefined in scenarios/"
    else:
        # load scenario from configuration file
        policy_dir = policy_path.parent
        logged_cfg_path = policy_dir / ".hydra" / "config.yaml"
        assert logged_cfg_path.is_file()
        logged_cfg = OmegaConf.load(logged_cfg_path)
        print("Configuration used to train loaded policy.")
        print(OmegaConf.to_yaml(logged_cfg))
        trained_scenario = logged_cfg.scenario

    return policy_weights, trained_scenario
