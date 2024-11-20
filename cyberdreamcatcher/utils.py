import inspect
from pathlib import Path
from bidict import bidict
import logging

import torch

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
