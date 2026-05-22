import gymnasium as gym
import copy

from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import (
    DefaultPPOTorchRLModule,
)
from ray.rllib.core import Columns
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import FLOAT_MIN


torch, _ = try_import_torch()


class ActionMaskRLMBase:
    def __init__(self, *args, observation_space=None, **kwargs):
        if not isinstance(observation_space, gym.spaces.Dict):
            raise ValueError("Action masking requires a gym.spaces.Dict observation space.")
        kwargs["observation_space"] = observation_space["observation"]
        super().__init__(*args, **kwargs)


class TorchActionMaskRLM(ActionMaskRLMBase, DefaultPPOTorchRLModule):
    def _forward_inference(self, batch, **kwargs):
        return mask_forward_fn_torch(super()._forward_inference, batch, **kwargs)

    def _forward_train(self, batch, *args, **kwargs):
        return mask_forward_fn_torch(super()._forward_train, batch, **kwargs)

    def _forward_exploration(self, batch, *args, **kwargs):
        return mask_forward_fn_torch(super()._forward_exploration, batch, **kwargs)

    def compute_values(self, batch, embeddings=None):
        batch = _strip_observation(batch)
        return super().compute_values(batch, embeddings=embeddings)


def mask_forward_fn_torch(forward_fn, batch, **kwargs):
    _check_batch(batch)
    batch = copy.copy(batch)
    action_mask = batch[SampleBatch.OBS]["action_mask"]
    batch[SampleBatch.OBS] = batch[SampleBatch.OBS]["observation"]
    outputs = forward_fn(batch, **kwargs)
    logits = outputs[SampleBatch.ACTION_DIST_INPUTS]
    inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
    outputs[SampleBatch.ACTION_DIST_INPUTS] = logits + inf_mask
    return outputs


def _strip_observation(batch):
    batch = copy.copy(batch)
    obs_key = Columns.OBS if Columns.OBS in batch else SampleBatch.OBS
    if isinstance(batch[obs_key], dict) and "observation" in batch[obs_key]:
        batch[obs_key] = batch[obs_key]["observation"]
    return batch


def _check_batch(batch):
    if "action_mask" not in batch[SampleBatch.OBS]:
        raise ValueError("Observation is missing action_mask.")
    if "observation" not in batch[SampleBatch.OBS]:
        raise ValueError("Observation is missing observation payload.")
