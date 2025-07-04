import random
import inspect
from pathlib import Path
from bidict import bidict
import logging
import sys
import comet_ml
import yaml

from omegaconf import OmegaConf
import numpy as np
import pandas as pd
import torch
from torch.profiler import profile, ProfilerActivity, record_function
from torch_geometric.data import Data, Batch

import CybORG

LOGGER = logging.getLogger(__name__)


def set_all_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_action_names(cyborg_action):
    "Converts action name to the equivalent gymnasium action index."
    action_name = cyborg_action.__class__.__name__
    host_name = getattr(cyborg_action, "hostname", None)
    return host_name, action_name


def instantiate_action(host_name, action_name, agent_name="Blue"):
    """Create instantiate the class object with the given host."""

    action_class = getattr(CybORG.Shared.Actions, action_name)

    if action_name == "Sleep":
        action = action_class()
    elif action_name == "Monitor":
        action = action_class(session=0, agent=agent_name)
    else:
        action = action_class(session=0, agent=agent_name, hostname=host_name)
    return action


def count_parameters(model, submodule=None):
    if submodule:
        assert isinstance(submodule, str), "Please provide the name of the submodule."
        return sum(p.numel() for p in model.get_submodule(submodule).parameters())
    return sum(p.numel() for p in model.parameters())


def gradient_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm


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
            raise ValueError(
                f"Vector is too short to populate state_dict. "
                f"Needed {current_pos + num_elements} elements, but vector has {len(vector)}. "
                f"Issue with key '{k}'."
            )
        chunk = vector[current_pos : current_pos + num_elements]
        # Ensure chunk has the correct number of elements before reshaping
        if chunk.size != num_elements:
            raise ValueError(
                f"Shape mismatch for key '{k}'. "
                f"Expected {num_elements} elements, but got {chunk.size} from vector slice."
            )
        new_state_dict[k] = (
            torch.from_numpy(chunk)
            .reshape(shape)
            .to(v_template.device)
            .type(v_template.dtype)
        )
        current_pos += num_elements
    # Final check that the entire vector was used
    if current_pos != len(vector):
        raise ValueError(
            f"Vector length ({len(vector)}) does not match the total number "
            f"of elements in the template state_dict ({current_pos}). "
            f"Vector might be too long."
        )
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
    scenarios = [file.stem for file in scenarios_dir.iterdir() if file.is_file()]
    return scenarios


def load_trained_weights(policy_weights_path, weights_only=True):
    """Load trained weights safely and extract scenario
    from logged training file if not specified"""

    policy_path = Path(policy_weights_path)
    assert policy_path.is_file()

    policy_weights = torch.load(policy_path, weights_only=weights_only)

    # load scenario from configuration file
    policy_dir = policy_path.parent

    # Try to load Hydra config first
    logged_cfg_path = policy_dir / ".hydra" / "config.yaml"

    logged_cfg = None
    if logged_cfg_path.is_file():
        logged_cfg = OmegaConf.load(logged_cfg_path)
        LOGGER.info(f"Loaded Hydra configuration from: {logged_cfg_path}")
    else:
        # If Hydra config not found, try to load a Comet-downloaded config
        comet_cfg_filename = policy_path.stem + "_comet_config.yaml"
        comet_cfg_path = policy_dir / comet_cfg_filename
        if comet_cfg_path.is_file():
            logged_cfg = OmegaConf.load(comet_cfg_path)
            LOGGER.info(f"Loaded Comet configuration from: {comet_cfg_path}")

    if logged_cfg:
        LOGGER.info("Configuration used to train loaded policy:")
        LOGGER.info(OmegaConf.to_yaml(logged_cfg))
    else:
        LOGGER.warning(
            f"Configuration file was not found for the policy weights: {policy_weights_path}."
            " Looked for a Hydra config (.hydra/config.yaml) and a Comet config (*_comet_config.yaml)."
        )

    return policy_weights, logged_cfg


def get_policy_weights_and_config(cfg, output_dir):
    """Loads policy weights and configuration from a local path or Comet ML.

    This function is a centralized utility to handle loading model weights. It can
    take weights from a local file path (`policy_weights`) or download them from
    a Comet ML experiment (`comet_experiment_key`). It ensures that these two
    options are mutually exclusive.

    Args:
        cfg: A configuration object (e.g., a Hydra Cfg dataclass) that contains
             parameters like `policy_weights`, `comet_experiment_key`,
             `comet_model_name`, and `comet_model_step`.
        output_dir: The directory where assets downloaded from Comet ML should be
                    saved.

    Returns:
        A tuple containing:
        - policy_weights: The loaded model weights (e.g., a state dictionary).
        - logged_cfg: The configuration that was logged with the weights.
        Returns (None, None) if no weights are loaded.
    """
    assert not (
        cfg.policy_weights and cfg.comet_experiment_key
    ), "Provide either 'policy_weights' or 'comet_experiment_key', not both."

    policy_path = None
    if cfg.comet_experiment_key:
        policy_path = download_model_from_comet(
            experiment_key=cfg.comet_experiment_key,
            asset_filename=cfg.comet_model_name,
            output_dir=output_dir,
            step=cfg.comet_model_step,
        )
    elif cfg.policy_weights:
        policy_path = cfg.policy_weights

    policy_weights, logged_cfg = None, None
    if policy_path:
        policy_weights, logged_cfg = load_trained_weights(
            policy_path, weights_only=False
        )

    return policy_weights, logged_cfg


def download_model_from_comet(
    experiment_key: str, asset_filename: str, output_dir: Path, step: int | None = None
):
    """Downloads a model asset from a Comet ML experiment. Also downloads the
    experiment's hyperparameters and saves them to a YAML file in the same
    directory as the model.

    Args:
        experiment_key: The key of the Comet ML experiment.
        asset_filename: The filename of the asset to download. Can include path components.
        output_dir: The directory to save the downloaded asset.
        step: The step at which the asset was logged.
    """
    LOGGER.info(
        f"Attempting to download asset '{asset_filename}' from experiment '{experiment_key}' for step {step}."
    )
    api = comet_ml.API()
    try:
        experiment = api.get_experiment_by_key(experiment_key)
    except Exception:
        LOGGER.error(f"Experiment {experiment_key} not found.")
        return

    # Determine the final output path for the model asset
    model_output_path = Path(output_dir) / asset_filename

    # Download and save hyperparameters
    params = experiment.get_parameters_summary()
    params_dict = {p["name"]: p.get("valueCurrent") for p in params}

    if params_dict:
        # Config filename is based on the model's filename (not the full asset path)
        config_filename = Path(model_output_path.name).stem + "_comet_config.yaml"
        # Config is saved in the same directory as the model
        config_path = model_output_path.parent / config_filename
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            yaml.dump(params_dict, f, indent=4)
        LOGGER.info(f"Hyperparameters saved to {config_path}")
    else:
        LOGGER.warning(
            f"Could not find any parameters for experiment {experiment_key}. Hyperparameters not saved."
        )

    asset_list = experiment.get_asset_list()

    found_assets = []
    for asset in asset_list:
        match_step = step is None or asset.get("step") == step
        match_name = asset.get("fileName") == asset_filename
        if match_step and match_name:
            found_assets.append(asset)

    LOGGER.info(f"Found {len(found_assets)} matching assets.")
    if not found_assets:
        msg = f"Asset '{asset_filename}' not found in experiment {experiment_key}"
        if step is not None:
            msg += f" for step {step}"
        LOGGER.error(msg + ".")

        if asset_list:
            LOGGER.info("Available assets in the experiment:")
            # Create a formatted list of assets for better readability
            assets_info = [
                f"  - Filename: {a.get('fileName')}, Step: {a.get('step')}, Size: {a.get('size')}"
                for a in asset_list
            ]
            LOGGER.info("\n".join(assets_info))
        else:
            LOGGER.info("No assets found in the experiment.")
        LOGGER.info("Halting execution.")
        sys.exit(1)

    if len(found_assets) > 1:
        LOGGER.warning(
            f"Multiple assets found for '{asset_filename}' and step {step}. Using the first one."
        )

    asset_to_download = found_assets[0]
    asset_id = asset_to_download["assetId"]

    LOGGER.info(
        f"Downloading asset '{asset_filename}' (step: {step}) from experiment {experiment_key}..."
    )
    asset_data = experiment.get_asset(asset_id, return_type="binary")

    # Use the path defined earlier
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_output_path, "wb") as f:
        f.write(asset_data)

    LOGGER.info(f"Asset saved to {model_output_path}")
    return model_output_path


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

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
        record_shapes=True,
    ) as prof:
        with record_function("model_inference"):
            _ = model(data)

    LOGGER.info(
        prof.key_averages().table(
            sort_by="cuda_time_total" if device == "cuda" else "cpu_time_total"
        )
    )


def get_scenario(name="Scenario2", from_cyborg=True):
    if from_cyborg:
        # scenario_path = inspect.getfile(CybORG)[:-10] + f"/Shared/Scenarios/{self.scenario}.yaml"
        cyborg_path = Path(inspect.getfile(CybORG)).resolve()
        scenario_dir = cyborg_path.parent / "Shared" / "Scenarios"
    else:
        scenario_dir = Path(__file__).resolve().parent.parent / "scenarios"

    scenario_path = scenario_dir / Path(name).with_suffix(".yaml")

    assert scenario_path.exists()

    return scenario_path
