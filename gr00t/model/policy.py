# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError

from gr00t.data.dataset import ModalityConfig
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.schema import DatasetMetadata
from gr00t.data.transform.base import ComposedModalityTransform
from gr00t.model.gr00t_n1 import GR00T_N1_5

COMPUTE_DTYPE = torch.bfloat16


class BasePolicy(ABC):
    @abstractmethod
    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Abstract method to get the action for a given state.

        Args:
            observations: The observations from the environment.

        Returns:
            The action to take in the environment in dictionary format.
        """
        raise NotImplementedError

    @abstractmethod
    def get_modality_config(self) -> Dict[str, ModalityConfig]:
        """
        Return the modality config of the policy.
        """
        raise NotImplementedError


class Gr00tPolicy(BasePolicy):
    """
    A wrapper for Gr00t model checkpoints that handles loading the model, applying transforms,
    making predictions, and unapplying transforms. This loads some custom configs, stats
    and metadata related to the model checkpoints used
    in the Gr00t model.
    """

    def __init__(
        self,
        model_path: str,
        embodiment_tag: Union[str, EmbodimentTag],
        modality_config: Dict[str, ModalityConfig],
        modality_transform: ComposedModalityTransform,
        denoising_steps: Optional[int] = None,
        device: Union[int, str] = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the Gr00tPolicy.

        Args:
            model_path (str): Path to the model checkpoint directory or the huggingface hub id.
            modality_config (Dict[str, ModalityConfig]): The modality config for the model.
            modality_transform (ComposedModalityTransform): The modality transform for the model.
            embodiment_tag (Union[str, EmbodimentTag]): The embodiment tag for the model.
            denoising_steps: Number of denoising steps to use for the action head.
            device (Union[int, str]): Device to run the model on.
        """
        try:
            # NOTE(YL) this returns the local path to the model which is normally
            # saved in ~/.cache/huggingface/hub/
            model_path = snapshot_download(model_path, repo_type="model")
            # HFValidationError, RepositoryNotFoundError
        except (HFValidationError, RepositoryNotFoundError):
            print(
                f"Model not found or avail in the huggingface hub. Loading from local path: {model_path}"
            )

        self._modality_config = modality_config
        self._modality_transform = modality_transform
        self._modality_transform.eval()  # set this to eval mode
        self.model_path = Path(model_path)
        self.device = device

        # Convert string embodiment tag to EmbodimentTag enum if needed
        if isinstance(embodiment_tag, str):
            self.embodiment_tag = EmbodimentTag(embodiment_tag)
        else:
            self.embodiment_tag = embodiment_tag

        # Load model
        self._load_model(model_path)
        # Load transforms
        self._load_metadata(self.model_path / "experiment_cfg")
        # Load horizons
        self._load_horizons()

        if denoising_steps is not None:
            if hasattr(self.model, "action_head") and hasattr(
                self.model.action_head, "num_inference_timesteps"
            ):
                self.model.action_head.num_inference_timesteps = denoising_steps
                print(f"Set action denoising steps to {denoising_steps}")

    def apply_transforms(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply transforms to the observation.

        Args:
            obs (Dict[str, Any]): The observation to transform.

        Returns:
            Dict[str, Any]: The transformed observation.
        """
        # Ensure correct dimensions before applying transforms
        return self._modality_transform(obs)

    def unapply_transforms(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Unapply transforms to the action.

        Args:
            action (Dict[str, Any]): The action to unapply transforms to.

        Returns:
            Dict[str, Any]: The untransformed action.
        """
        return self._modality_transform.unapply(action)

    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a prediction with the model.
        Args:
            obs (Dict[str, Any]): The observation to make a prediction for.

        e.g. obs = {
            "video.<>": np.ndarray,  # (T, H, W, C)
            "state.<>": np.ndarray, # (T, D)
            "annotation.<>": np.ndarray, # (T, )
        }

        or with batched input:
        e.g. obs = {
            "video.<>": np.ndarray,, # (B, T, H, W, C)
            "state.<>": np.ndarray, # (B, T, D)
            "annotation.<>": np.ndarray, # (B, T, )
        }

        Returns:
            Dict[str, Any]: The predicted action.
        """
        # let the get_action handles both batch and single input
        is_batch = self._check_state_is_batched(observations)
        if not is_batch:
            observations = unsqueeze_dict_values(observations)

        # NOTE(YL): ensure keys are all in numpy array
        for k, v in observations.items():
            if not isinstance(v, np.ndarray):
                observations[k] = np.array(v)

        # Apply transforms
        normalized_input = self.apply_transforms(observations)

        normalized_action = self._get_action_from_normalized_input(normalized_input)
        unnormalized_action = self._get_unnormalized_action(normalized_action)
        if not is_batch:
            unnormalized_action = squeeze_dict_values(unnormalized_action)
        return unnormalized_action

    def prepare_action_prev(self, data: dict):
        """
        Pad to max_action_dim, return masks.
        """

        actions = data[0]
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        assert actions.shape[0] == self.model.action_head.action_horizon, f"{actions.shape=}, {self.model.action_head.action_horizon=}"

        n_action_tokens = actions.shape[0]  # T
        n_action_dims = actions.shape[1]

        assert (
            n_action_dims <= self.model.action_head.action_dim
        ), f"Action dim {n_action_dims} exceeds max allowed {self.model.action_head.action_dim}."

        # Pad the channel dimension
        actions = np.pad(actions, ((0, 0), (0, self.model.action_head.action_dim - n_action_dims)), "constant")

        # Create mask: [T, max_action_dim]
        actions_mask = np.zeros((n_action_tokens, self.model.action_head.action_dim), dtype=bool)
        actions_mask[:, :n_action_dims] = True

        return actions, actions_mask, n_action_tokens


    def get_action_rtc(self, observations: Dict[str, Any], rtc: Dict[str, Any]) -> Dict[str, Any]:
        # 1) 観測は通常どおり正規化（video/state/text）
        is_batch = self._check_state_is_batched(observations)
        if not is_batch:
            observations = unsqueeze_dict_values(observations)
        obs_np = {k: (v if isinstance(v, np.ndarray) else np.array(v)) for k, v in observations.items()}
        normalized_input = self.apply_transforms(obs_np)   # ← Transform は観測のみ

        # 2) 前チャンクは action 統計で手動正規化して合流
        if rtc.get("rtc_prev_action", None) is not None:
            prev = rtc["rtc_prev_action"]  # 非正規化 [H,D] or [B,H,D]
            # # 必要なら右パディング（推奨：先に呼び出し側で H に揃える）
            # H = self.model.action_head.action_horizon
            # if isinstance(prev, np.ndarray) and prev.ndim == 2 and prev.shape[0] != H:
            #     if prev.shape[0] < H:
            #         pad = np.zeros((H - prev.shape[0], prev.shape[1]), dtype=prev.dtype)
            #         prev = np.concatenate([prev, pad], axis=0)
            #     else:
            #         prev = prev[:H]
            prev_norm = self._normalize_prev_action_with_stats(prev)  # torch [B,H,D]


            actions, actions_mask, n_action_tokens = self.prepare_action_prev(prev_norm)

            adtype = self.model.action_head.dtype if hasattr(self.model.action_head, "dtype") else torch.float32
            dev    = self.device

            A_prev_t = torch.as_tensor(actions, device=dev, dtype=adtype).unsqueeze(0)

            normalized_input["rtc_prev_action"] = A_prev_t

        # 3) 付帯（W, beta, clip, angle idx）を Tensor 化して追加
        device = self.device
        dtype  = (self.model.action_head.dtype
                if hasattr(self.model.action_head, "dtype") else torch.float32)

        if rtc.get("rtc_weight_mask", None) is not None:
            W = rtc["rtc_weight_mask"]
            normalized_input["rtc_weight_mask"] = (
                W if isinstance(W, torch.Tensor) else torch.as_tensor(W, device=device, dtype=dtype)
            )

        if rtc.get("rtc_beta", None) is not None:
            beta = rtc["rtc_beta"]
            normalized_input["rtc_beta"] = (
                beta if isinstance(beta, torch.Tensor) else torch.tensor(float(beta), device=device, dtype=dtype)
            )

        if rtc.get("rtc_guidance_clip", None) is not None:
            clip = rtc["rtc_guidance_clip"]
            normalized_input["rtc_guidance_clip"] = (
                clip if isinstance(clip, torch.Tensor) else torch.tensor(float(clip), device=device, dtype=dtype)
            )

        if rtc.get("rtc_angle_indices", None) is not None:
            idx = rtc["rtc_angle_indices"]
            normalized_input["rtc_angle_indices"] = (
                idx if isinstance(idx, torch.Tensor) else torch.as_tensor(idx, device=device, dtype=torch.long)
            )

        # 4) モデル実行（RTC対応パス）
        normalized_action = self._get_action_rtc_from_normalized_input(normalized_input)
        unnormalized_action = self._get_unnormalized_action(normalized_action)
        if not is_batch:
            unnormalized_action = squeeze_dict_values(unnormalized_action)
        return unnormalized_action



    def _get_action_from_normalized_input(self, normalized_input: Dict[str, Any]) -> torch.Tensor:
        # Set up autocast context if needed
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
            model_pred = self.model.get_action(normalized_input)

        normalized_action = model_pred["action_pred"].float()
        return normalized_action

    def _get_action_rtc_from_normalized_input(self, normalized_input: Dict[str, Any]) -> torch.Tensor:
        # Set up autocast context if needed
        with torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
            model_pred = self.model.get_action_rtc(normalized_input)

        normalized_action = model_pred["action_pred"].float()
        return normalized_action

    def _get_unnormalized_action(self, normalized_action: torch.Tensor) -> Dict[str, Any]:
        return self.unapply_transforms({"action": normalized_action.cpu()})

    def get_modality_config(self) -> Dict[str, ModalityConfig]:
        """
        Get the modality config for the model, overrides the base class method
        """
        return self._modality_config

    @property
    def modality_config(self) -> Dict[str, ModalityConfig]:
        return self._modality_config

    @property
    def modality_transform(self) -> ComposedModalityTransform:
        return self._modality_transform

    @property
    def video_delta_indices(self) -> np.ndarray:
        """Get the video delta indices."""
        return self._video_delta_indices

    @property
    def state_delta_indices(self) -> np.ndarray | None:
        """Get the state delta indices."""
        return self._state_delta_indices

    @property
    def denoising_steps(self) -> int:
        """Get the number of denoising steps."""
        return self.model.action_head.num_inference_timesteps

    @denoising_steps.setter
    def denoising_steps(self, value: int):
        """Set the number of denoising steps."""
        self.model.action_head.num_inference_timesteps = value

    def _check_state_is_batched(self, obs: Dict[str, Any]) -> bool:
        for k, v in obs.items():
            if "state" in k and len(v.shape) < 3:  # (B, Time, Dim)
                return False
        return True

    def _load_model(self, model_path):
        model = GR00T_N1_5.from_pretrained(model_path, torch_dtype=COMPUTE_DTYPE)
        model.eval()  # Set model to eval mode
        model.to(device=self.device)  # type: ignore

        # Update action_horizon to match modality config
        # Get the expected action horizon from the modality config
        expected_action_horizon = len(self._modality_config["action"].delta_indices)

        if expected_action_horizon != model.action_head.config.action_horizon:
            print(
                f"Policy: Recreating action head with action_horizon {expected_action_horizon} (was {model.action_head.config.action_horizon})"
            )

            # Update the action head config
            new_action_head_config = model.action_head.config
            new_action_head_config.action_horizon = expected_action_horizon

            # Import the FlowmatchingActionHead class
            from gr00t.model.action_head.flow_matching_action_head import (
                FlowmatchingActionHead,
            )

            # Create new action head with updated config
            new_action_head = FlowmatchingActionHead(new_action_head_config)

            # Copy the weights from the old action head to the new one
            new_action_head.load_state_dict(model.action_head.state_dict(), strict=False)

            # Replace the action head
            model.action_head = new_action_head

            # Update model config AND the action_head_cfg dictionary that gets saved
            model.config.action_horizon = expected_action_horizon
            model.action_horizon = expected_action_horizon
            model.config.action_head_cfg["action_horizon"] = expected_action_horizon

        model.action_head.to(device=self.device, dtype=COMPUTE_DTYPE)

        # LayerNorm 系を最後にもう一度強制で揃える（保険）
        for m in model.modules():
            if isinstance(m, torch.nn.LayerNorm):
                m.to(device=self.device, dtype=COMPUTE_DTYPE)


        self.model = model

    def _load_metadata(self, exp_cfg_dir: Path):
        """Load the transforms for the model."""
        # Load metadata for normalization stats
        metadata_path = exp_cfg_dir / "metadata.json"
        with open(metadata_path, "r") as f:
            metadatas = json.load(f)

        # Get metadata for the specific embodiment
        metadata_dict = metadatas.get(self.embodiment_tag.value)
        if metadata_dict is None:
            raise ValueError(
                f"No metadata found for embodiment tag: {self.embodiment_tag.value}",
                f"make sure the metadata.json file is present at {metadata_path}",
            )

        metadata = DatasetMetadata.model_validate(metadata_dict)

        self._modality_transform.set_metadata(metadata)
        self.metadata = metadata

    def _load_horizons(self):
        """Load the horizons needed for the model."""
        # Get modality configs
        # Video horizons
        self._video_delta_indices = np.array(self._modality_config["video"].delta_indices)
        self._assert_delta_indices(self._video_delta_indices)
        self._video_horizon = len(self._video_delta_indices)
        # State horizons (if used)
        if "state" in self._modality_config:
            self._state_delta_indices = np.array(self._modality_config["state"].delta_indices)
            self._assert_delta_indices(self._state_delta_indices)
            self._state_horizon = len(self._state_delta_indices)
        else:
            self._state_horizon = None
            self._state_delta_indices = None

    def _assert_delta_indices(self, delta_indices: np.ndarray):
        """Assert that the delta indices are valid."""
        # All delta indices should be non-positive because there's no way to get the future observations
        assert np.all(delta_indices <= 0), f"{delta_indices=}"
        # The last delta index should be 0 because it doesn't make sense to not use the latest observation
        assert delta_indices[-1] == 0, f"{delta_indices=}"
        if len(delta_indices) > 1:
            # The step is consistent
            assert np.all(
                np.diff(delta_indices) == delta_indices[1] - delta_indices[0]
            ), f"{delta_indices=}"
            # And the step is positive
            assert (delta_indices[1] - delta_indices[0]) > 0, f"{delta_indices=}"

    @torch.no_grad()
    def build_rtc_weight_mask(self, H, d, s, lam=0.3, B=1, D=None, device="cuda", dtype=torch.float32):
        """
        先頭[0..d-1]=1, 中間[d..H-s-1]=exp(-lam*(t-d)), 末尾[H-s..H-1]=0
        返り: [B,H,1] または [B,H,D]
        """
        d = torch.full((B,), int(d), device=device, dtype=torch.long)
        s = torch.full((B,), int(s), device=device, dtype=torch.long)
        d = d.clamp(min=0, max=H); s = torch.minimum(s, (H-d).clamp(min=0))
        t = torch.arange(H, device=device).view(1, H, 1)
        d_ = d.view(B,1,1); s_ = s.view(B,1,1)
        freeze = (t < d_).float()
        free   = (t >= (H - s_)).float()
        mid    = (1.0 - freeze) * (1.0 - free)
        k = (t - d_).clamp(min=0)
        w_mid = torch.exp(-lam * k) * mid
        W = (freeze + w_mid)  # 末尾は0のまま
        if D is not None:
            W = W.expand(B, H, D)
        return W.to(dtype=dtype)

    def get_action_rel_stats_tensors(self):
        # DatasetStatisticalValues を取得（relative）
        rel = self.metadata.statistics.action["relative"]  # or: self.metadata.statistics.action.relative

        mean = torch.as_tensor(rel.mean, device=self.device, dtype=torch.float32).view(1, 1, -1)
        std  = torch.as_tensor(rel.std,  device=self.device, dtype=torch.float32).clamp_min(1e-6).view(1, 1, -1)

        return mean, std




    def _normalize_prev_action_with_stats(self, prev):
        """
        prev: [H,D] or [B,H,D] in *unnormalized (world)* space
        returns: torch.Tensor [B,H,D] in *normalized* space
        """
        if isinstance(prev, np.ndarray):
            prev = torch.from_numpy(prev)
        if not isinstance(prev, torch.Tensor):
            prev = torch.as_tensor(prev)

        if prev.ndim == 2:  # [H,D] -> [1,H,D]
            prev = prev.unsqueeze(0)

        mean, std = self.get_action_rel_stats_tensors()
        prev = prev.to(device=self.device, dtype=torch.float32)
        prev_norm = (prev - mean) / std
        return prev_norm



#######################################################################################################


# Helper functions
def unsqueeze_dict_values(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unsqueeze the values of a dictionary.
    This converts the data to be batched of size 1.
    """
    unsqueezed_data = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            unsqueezed_data[k] = np.expand_dims(v, axis=0)
        elif isinstance(v, list):
            unsqueezed_data[k] = np.array(v)
        elif isinstance(v, torch.Tensor):
            unsqueezed_data[k] = v.unsqueeze(0)
        else:
            unsqueezed_data[k] = v
    return unsqueezed_data


def squeeze_dict_values(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Squeeze the values of a dictionary. This removes the batch dimension.
    """
    squeezed_data = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            squeezed_data[k] = np.squeeze(v)
        elif isinstance(v, torch.Tensor):
            squeezed_data[k] = v.squeeze()
        else:
            squeezed_data[k] = v
    return squeezed_data
