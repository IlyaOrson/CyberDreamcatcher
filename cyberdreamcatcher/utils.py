import random
import inspect
from pathlib import Path
from bidict import bidict
import logging

from omegaconf import OmegaConf
import numpy as np
import pandas as pd
import torch
from torch.profiler import profile, ProfilerActivity, record_function
from torch_geometric.data import Data

import CybORG

LOGGER = logging.getLogger(__name__)

def set_all_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Add these helper functions for parameter conversion
def state_dict_to_vector(state_dict: dict) -> np.ndarray:
    """Converts a PyTorch state_dict to a flat NumPy array."""
    return np.concatenate([v.cpu().numpy().flatten() for v in state_dict.values()])

def vector_to_state_dict(vector: np.ndarray, template_state_dict: dict) -> dict:
    """Converts a flat NumPy array back to a PyTorch state_dict using a template."""
    new_state_dict = {}
    current_pos = 0
    for k, v_template in template_state_dict.items():
        shape = v_template.shape
        num_elements = v_template.numel()
        # Ensure slice does not exceed vector bounds
        if current_pos + num_elements > len(vector):
             raise ValueError(f"Vector is too short to populate state_dict. "
                              f"Needed {current_pos + num_elements} elements, but vector has {len(vector)}. "
                              f"Issue with key '{k}'.")
        chunk = vector[current_pos : current_pos + num_elements]
        # Ensure chunk has the correct number of elements before reshaping
        if chunk.size != num_elements:
            raise ValueError(f"Shape mismatch for key '{k}'. "
                             f"Expected {num_elements} elements, but got {chunk.size} from vector slice.")
        new_state_dict[k] = torch.from_numpy(chunk).reshape(shape).to(v_template.device).type(v_template.dtype)
        current_pos += num_elements
    # Final check that the entire vector was used
    if current_pos != len(vector):
         raise ValueError(f"Vector length ({len(vector)}) does not match the total number "
                          f"of elements in the template state_dict ({current_pos}). "
                          f"Vector might be too long.")
    return new_state_dict


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


def long_format_dataframe(stacked_rewards_to_go):
    df = pd.DataFrame(stacked_rewards_to_go)
    return df.melt(var_name="timestep", value_name="reward_to_go")


def downsample_dataframe(df_long, step=None, steps=None):
    assert step or steps
    assert not (step and steps)
    if steps:
        return df_long.query(f"timestep in {steps}")
    elif step:
        last_timestep = df_long["timestep"].max()
        df_down_sample = df_long.query(f"timestep % {step} == 0")
        df_last_timestep = df_long.query(f"timestep == {last_timestep}")
        return pd.concat([df_down_sample, df_last_timestep])
    else:
        return df_long


def profile_inference(data: Data, model: torch.nn.Module, device: str):
    model = model.to(device)
    data = data.to(device)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                profile_memory=True,
                record_shapes=True) as prof:
        with record_function("model_inference"):
            _ = model(data)

    print(prof.key_averages().table(sort_by="cuda_time_total" if device == 'cuda' else "cpu_time_total"))
