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

import random
import re
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import tree
from einops import rearrange
from PIL import Image
from pydantic import Field, PrivateAttr
from transformers import AutoProcessor, ProcessorMixin
from transformers.data.data_collator import DataCollatorMixin
from transformers.feature_extraction_utils import BatchFeature

from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING, EmbodimentTag
from gr00t.data.schema import DatasetMetadata
from gr00t.data.transform.base import InvertibleModalityTransform

from .backbone.eagle_backbone import DEFAULT_EAGLE_PATH


def formalize_language(language: str) -> str:
    """
    1. Force lowercase
    2. Remove all punctuations
    """
    language = language.lower()
    language = re.sub(r"[^\w\s]", "", language)
    return language


def build_eagle_processor(eagle_path: str) -> ProcessorMixin:
    eagle_processor = AutoProcessor.from_pretrained(
        eagle_path, trust_remote_code=True, use_fast=True
    )
    eagle_processor.tokenizer.padding_side = "left"
    
    eagle_processor.model_max_length = 64
    eagle_processor.tokenizer.truncation_side = "right"
    eagle_processor.tokenizer.pad_token = eagle_processor.tokenizer.eos_token
    return eagle_processor

# def build_eagle_processor(eagle_path: str) -> ProcessorMixin:
#     p = AutoProcessor.from_pretrained(eagle_path, trust_remote_code=True, use_fast=True)
#     p.tokenizer.padding_side = "left"

#     # 追加：常に max_length で詰める
#     # 適切な長さを設定（例：512）。モデルに合わせて。
#     p.tokenizer.model_max_length = 32
#     p.tokenizer.truncation_side = "right"
#     p.tokenizer.pad_token = p.tokenizer.eos_token
#     p.feature_extractor.padding = "max_length" if hasattr(p, "feature_extractor") else None
#     return p

def _pin_if_cpu_tensor(x):
    # CPU テンソルは pinned に。GPU テンソル/非テンソルはそのまま返す
    if isinstance(x, torch.Tensor) and x.device.type == "cpu":
        try:
            return x.pin_memory()
        except RuntimeError:
            return x
    return x

def _to_np_float32(x):
    # torch.Tensor → CPU float32 numpy
    if isinstance(x, torch.Tensor):
        # dataloader worker 内なら大抵 CPU tensor のはずだが、念のため detach+cpu
        x = x.detach().cpu().to(torch.float32)
        return x.numpy()
    # numpy / list → numpy float32
    arr = np.asarray(x)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    return arr

def _to_np_bool(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().to(torch.bool)
        return x.numpy()
    arr = np.asarray(x)
    if arr.dtype != np.bool_:
        arr = arr.astype(np.bool_, copy=False)
    return arr


def collate(features: List[dict], eagle_processor, training) -> dict:
    batch = {}
    keys = features[0].keys()

    for key in keys:
        values = [elem[key] for elem in features]

        if key == "eagle_content":
            text_list = []
            if training:
                text_list_out = []
            else:
                text_list_out = None
            image_inputs = []
            for v in values:
                curr_text_list = v["text_list"]
                if training:
                    curr_text_list_out = v["text_list_out"]
                curr_image_inputs = v["image_inputs"]
                text_list += curr_text_list
                if training:
                    text_list_out += curr_text_list_out
                image_inputs += curr_image_inputs
            
            #print(text_list)
            #print(text_list_out)
            eagle_inputs = eagle_processor(
                text=text_list, target_text=text_list_out, images=image_inputs, return_tensors="pt", padding=True
            )
            # eagle_inputs = eagle_processor(
            #     text=text_list,
            #     images=image_inputs,
            #     return_tensors="pt",
            #     padding="max_length",     # ← ここを True ではなく "max_length"
            #     truncation=True,          # ← 端数が来ても切る
            #     max_length=eagle_processor.tokenizer.model_max_length,
            # )
            for k, v in eagle_inputs.items():
                k = "eagle_" + k
                batch[k] = v
                #batch[k] = _pin_if_cpu_tensor(v)
        elif key in ("pixel_values", "image_grid_thw", "attention_mask", "input_ids"):
            # Concat in existing batch dimension.
            batch[key] = torch.cat(values)
            #batch[key] = _pin_if_cpu_tensor(torch.cat(values))
        else:
            # state, state_mask, action and action_mask.
            # Stack to form the batch dimension.
            batch[key] = torch.from_numpy(np.stack(values))
            #batch[key] = _pin_if_cpu_tensor(torch.from_numpy(np.stack(values)))
            # arr = np.stack(values)

            # # dtype 正規化：float64 禁止、float32 に揃える
            # if arr.dtype == np.float64:
            #     arr = arr.astype(np.float32, copy=False)

            # # mask らしきものは bool を強制
            # if key.endswith("_mask") and arr.dtype != np.bool_:
            #     arr = arr.astype(np.bool_, copy=False)

            # # そのほかの数値は float32 / int 系はそのまま
            # t = torch.from_numpy(arr)
            # batch[key] = _pin_if_cpu_tensor(t)

    return batch
    #return tree.map_structure(_pin_if_cpu_tensor, batch)

class DefaultDataCollator(DataCollatorMixin):
    def __init__(self, eagle_path: str = DEFAULT_EAGLE_PATH):
        super().__init__()
        self.eagle_processor = build_eagle_processor(eagle_path)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        return collate(features, self.eagle_processor, True)


class GR00TTransform(InvertibleModalityTransform):

    # -- We inherit from ModalityTransform, so we keep apply_to as well --
    apply_to: list[str] = Field(
        default_factory=list, description="Not used in this transform, kept for compatibility."
    )
    training: bool = Field(
        default=True, description="Whether to apply the transform in training mode."
    )
    formalize_language: bool = Field(default=False, description="Formalize language if True.")
    embodiment_tag_mapping: dict[str, int] = Field(
        description="The projector index of each embodiment tag.",
        default=EMBODIMENT_TAG_MAPPING,
    )
    language_dropout_prob: float = Field(
        default=0.0,
        description="Dropout probability for language.",
    )

    # Private attributes to keep track of shapes/dimensions across apply/unapply
    _language_key: Optional[list[str]] = PrivateAttr(default=None)

    eagle_processor: ProcessorMixin = Field(default=build_eagle_processor(DEFAULT_EAGLE_PATH))

    # XEmbDiT arguments
    default_instruction: str = Field(default="Perform the default behavior.")
    default_reasoning: str = Field(default="The behavior is performing well.")
    max_state_dim: int
    max_action_dim: int
    state_horizon: int
    action_horizon: int

    max_length: int = 512
    embodiment_tag: EmbodimentTag | None = None


    def set_metadata(self, dataset_metadata: DatasetMetadata):
        """Set the metadata for the transform."""
        super().set_metadata(dataset_metadata)
        self.embodiment_tag = dataset_metadata.embodiment_tag

    def get_embodiment_tag(self) -> int:
        """Get the embodiment tag from the data."""
        assert (
            self.embodiment_tag is not None
        ), "Embodiment tag not set. Please call set_metadata first."
        return self.embodiment_tag_mapping[self.embodiment_tag.value]

    def check_keys_and_batch_size(self, data):
        grouped_keys = {}
        for key in data.keys():
            if "annotation" in key:
                modality = "language"
            else:
                try:
                    modality, _ = key.split(".")
                except:  # noqa: E722
                    modality = "others"  # will contain the video, state, and action
            if modality not in grouped_keys:
                grouped_keys[modality] = []
            grouped_keys[modality].append(key)
        # Use video key to determine batch size.
        video_ndim = data["video"].ndim
        if video_ndim == 5:  # Interpret as [T, V, H, W, C]
            is_batched = False
            batch_size = 1
        elif video_ndim == 6:  # Interpret as [B, T, V, H, W, C]
            is_batched = True
            batch_size = data["video"].shape[0]
        else:
            raise ValueError(f"Unsupported video number of dimensions: {video_ndim}")

        # Handle language
        if "language" in grouped_keys:
            #language_keys = grouped_keys["language"]
            language_keys = ["annotation.human.task_description", "annotation.reason.subtask_description"]
            assert len(language_keys) == 2, f"{language_keys=}"
            self._language_key = language_keys
        return is_batched, batch_size

    def _apply_vlm_processing(self, batch: dict) -> BatchFeature:
        """
        Args:
            batch:
                video: [V, T, C, H, W]
        Returns: required input with the format `BatchFeature`
        """
        # TODO(YL, FH): check if this is correct
        images = batch["images"]  # [V, T, C, H, W]
        images.shape[0]

        np_images = rearrange(images, "v t c h w -> (t v) c h w")
        text_content = []

        # handle language
        lang = batch["language"]
        if isinstance(lang, list):
            lang = lang[0]
        text_content.append({"type": "text", "text": lang})

        text_content_out = []

        # handle language
        lang_out = batch["language_out"]
        if isinstance(lang_out, list):
            lang_out = lang_out[0]
        text_content_out.append({"type": "text", "text": lang_out})

        eagle_images = [Image.fromarray(np.transpose(v, (1, 2, 0))) for v in np_images]
        eagle_image = [{"type": "image", "image": img} for img in eagle_images]
        eagle_conversation = [
            {
                "role": "user",
                "content": eagle_image + text_content,
            }
        ]

        text_list = [
            self.eagle_processor.apply_chat_template(
                eagle_conversation, tokenize=False, add_generation_prompt=True
            )
        ]
        lang_out += "<|im_end|>\n"
        image_inputs, video_inputs = self.eagle_processor.process_vision_info(eagle_conversation)
        eagle_content = {
            "image_inputs": image_inputs,
            "video_inputs": video_inputs,
            "text_list": text_list,
            "text_list_out": [lang_out],
        }
        inputs = {}
        inputs["eagle_content"] = eagle_content
        return inputs

    def _prepare_video(self, data: dict):
        """Process, stack, and pad images from data['video']."""
        ## TODO(YL, FH): check if this is correct
        images = rearrange(
            data["video"],
            "t v h w c -> v t c h w",
        )
        return images

    def _prepare_language(self, data: dict):
        """Tokenize data['language'] (or default_instruction if missing)."""
        if self._language_key is not None:
            raw_language = data[self._language_key[0]]
            raw_language_out = self.default_reasoning
            if self.training:
                raw_language_out = data[self._language_key[1]]
            if isinstance(raw_language, list):
                raw_language = raw_language[0]
            if self.training:
                if isinstance(raw_language_out, list):
                    raw_language_out = raw_language_out[0]

            # Language dropout
            if self.training and self.language_dropout_prob > 1e-9:
                if random.random() < self.language_dropout_prob:
                    raw_language = self.default_instruction

                if random.random() < self.language_dropout_prob:
                    raw_language_out = self.default_reasoning

            

        else:
            raw_language = self.default_instruction
            raw_language_out = self.default_reasoning
        return raw_language, raw_language_out

    # def _prepare_state(self, data: dict):
    #     """
    #     Gathers final state from data['state'], then pads to max_state_dim.
    #     Return (state, state_mask, n_state_tokens).
    #     """
    #     if "state" not in data:
    #         state = np.zeros((self.state_horizon, self.max_state_dim))
    #         state_mask = np.zeros((self.state_horizon, self.max_state_dim), dtype=bool)
    #         n_state_tokens = self.state_horizon
    #         return state, state_mask, n_state_tokens

    #     state = data["state"]
    #     assert state.shape[0] == self.state_horizon, f"{state.shape=}, {self.state_horizon=}"

    #     n_state_dims = state.shape[-1]

    #     # Instead of asserting, just take the first max_state_dim dimensions if needed
    #     if n_state_dims > self.max_state_dim:
    #         state = state[:, : self.max_state_dim]
    #         n_state_dims = self.max_state_dim
    #     else:
    #         # Pad up to max_state_dim if smaller
    #         state = np.pad(state, ((0, 0), (0, self.max_state_dim - n_state_dims)), "constant")

    #     # Create mask for real state dims
    #     state_mask = np.zeros_like(state).astype(bool)
    #     state_mask[:, :n_state_dims] = True

    #     # We only have 1 "proprio" token to represent the entire state
    #     n_state_tokens = state.shape[0]
    #     return state, state_mask, n_state_tokens

    def _prepare_state(self, data: dict):
        if "state" not in data:
            state = np.zeros((self.state_horizon, self.max_state_dim), dtype=np.float32)
            state_mask = np.zeros((self.state_horizon, self.max_state_dim), dtype=bool)
            n_state_tokens = self.state_horizon
            return state, state_mask, n_state_tokens

        state = _to_np_float32(data["state"])
        assert state.shape[0] == self.state_horizon, f"{state.shape=}, {self.state_horizon=}"

        n_state_dims = state.shape[-1]
        if n_state_dims > self.max_state_dim:
            state = state[:, : self.max_state_dim]
            n_state_dims = self.max_state_dim
        else:
            # pad を入れても dtype は float32 のまま
            state = np.pad(state, ((0, 0), (0, self.max_state_dim - n_state_dims)), "constant")

        state_mask = np.zeros_like(state, dtype=bool)
        state_mask[:, :n_state_dims] = True

        n_state_tokens = state.shape[0]
        return state, state_mask, n_state_tokens



    # def _prepare_action(self, data: dict):
    #     """
    #     Pad to max_action_dim, return masks.
    #     """
    #     if "action" not in data:
    #         actions = np.zeros((self.action_horizon, self.max_action_dim))
    #         actions_mask = np.zeros((self.action_horizon, self.max_action_dim), dtype=bool)
    #         n_action_tokens = self.action_horizon
    #         return actions, actions_mask, n_action_tokens

    #     actions = data["action"]
    #     assert actions.shape[0] == self.action_horizon, f"{actions.shape=}, {self.action_horizon=}"

    #     n_action_tokens = actions.shape[0]  # T
    #     n_action_dims = actions.shape[1]

    #     assert (
    #         n_action_dims <= self.max_action_dim
    #     ), f"Action dim {n_action_dims} exceeds max allowed {self.max_action_dim}."

    #     # Pad the channel dimension
    #     actions = np.pad(actions, ((0, 0), (0, self.max_action_dim - n_action_dims)), "constant")

    #     # Create mask: [T, max_action_dim]
    #     actions_mask = np.zeros((n_action_tokens, self.max_action_dim), dtype=bool)
    #     actions_mask[:, :n_action_dims] = True

    #     return actions, actions_mask, n_action_tokens
    
    def _prepare_action(self, data: dict):
        if "action" not in data:
            actions = np.zeros((self.action_horizon, self.max_action_dim), dtype=np.float32)
            actions_mask = np.zeros((self.action_horizon, self.max_action_dim), dtype=bool)
            n_action_tokens = self.action_horizon
            return actions, actions_mask, n_action_tokens

        actions = _to_np_float32(data["action"])
        assert actions.shape[0] == self.action_horizon, f"{actions.shape=}, {self.action_horizon=}"

        n_action_tokens = actions.shape[0]
        n_action_dims = actions.shape[1]
        assert n_action_dims <= self.max_action_dim, \
            f"Action dim {n_action_dims} exceeds max allowed {self.max_action_dim}."

        actions = np.pad(actions, ((0, 0), (0, self.max_action_dim - n_action_dims)), "constant")
        actions_mask = np.zeros((n_action_tokens, self.max_action_dim), dtype=bool)
        actions_mask[:, :n_action_dims] = True

        return actions, actions_mask, n_action_tokens



    # def apply_single(self, data: dict) -> dict:
    #     transformed_data = {}

    #     # 1) Prepare video and language with vlm processing.
    #     images = self._prepare_video(data)
    #     images = images.astype(np.uint8)
    #     language = self._prepare_language(data)
    #     batch_data = {"images": images, "language": language}
    #     vlm_outputs = self._apply_vlm_processing(batch_data)

    #     # 2) Prepare state
    #     state, state_mask, _ = self._prepare_state(data)
    #     transformed_data["state"] = state
    #     transformed_data["state_mask"] = state_mask

    #     if self.training:
    #         # 3) Prepare actions
    #         transformed_data["segmentation_target"] = np.zeros((2,))
    #         transformed_data["segmentation_target_mask"] = np.zeros((1,))
    #         transformed_data["has_real_action"] = np.ones((), dtype=bool)
    #         actions, actions_mask, _ = self._prepare_action(data)
    #         transformed_data["action"] = actions
    #         transformed_data["action_mask"] = actions_mask

    #     for k, v in vlm_outputs.items():
    #         assert k not in transformed_data, f"Key {k} already exists in transformed_data."
    #         transformed_data[k] = v

    #     transformed_data["embodiment_id"] = self.get_embodiment_tag()

    #     if self.training:
    #         action_and_mask_keys = ["action", "action_mask"]
    #         assert all(
    #             transformed_data[key].shape == transformed_data["action"].shape
    #             for key in action_and_mask_keys
    #         ), f"Shape mismatch: {[(key, transformed_data[key].shape) for key in action_and_mask_keys]}"

    #     return transformed_data

    def apply_single(self, data: dict) -> dict:
        transformed_data = {}

        # video はそのまま
        images = self._prepare_video(data)
        images = images.astype(np.uint8)

        language, language_out = self._prepare_language(data)
        batch_data = {"images": images, "language": language, "language_out": language_out}
        vlm_outputs = self._apply_vlm_processing(batch_data)

        # state
        state, state_mask, _ = self._prepare_state(data)
        transformed_data["state"] = state
        transformed_data["state_mask"] = state_mask

        if self.training:
            # --- ここ dtype 明示（NEW） ---
            transformed_data["segmentation_target"] = np.zeros((2,), dtype=np.float32)
            transformed_data["segmentation_target_mask"] = np.zeros((1,), dtype=bool)
            transformed_data["has_real_action"] = np.ones((), dtype=bool)
            # --------------------------------
            actions, actions_mask, _ = self._prepare_action(data)
            transformed_data["action"] = actions
            transformed_data["action_mask"] = actions_mask

        for k, v in vlm_outputs.items():
            assert k not in transformed_data, f"Key {k} already exists in transformed_data."
            transformed_data[k] = v

        transformed_data["embodiment_id"] = self.get_embodiment_tag()

        if self.training:
            action_and_mask_keys = ["action", "action_mask"]
            assert all(
                transformed_data[key].shape == transformed_data["action"].shape
                for key in action_and_mask_keys
            ), f"Shape mismatch: {[(key, transformed_data[key].shape) for key in action_and_mask_keys]}"

        return transformed_data


    def apply_batch(self, data: dict, batch_size: int) -> dict:
        # Split on batch dimension.
        data_split = [tree.map_structure(lambda x: x[i], data) for i in range(batch_size)]
        # Process each element.
        data_split_processed = [self.apply_single(elem) for elem in data_split]
        return collate(data_split_processed, self.eagle_processor, self.training)

    def apply(self, data: dict) -> dict:
        is_batched, batch_size = self.check_keys_and_batch_size(data)
        if is_batched:
            return self.apply_batch(data, batch_size)
        else:
            return self.apply_single(data)

    def unapply(self, data: dict) -> dict:
        # Leave as is so that ConcatTransform can split the values
        return data

    def unapply_reasoning(self, data):
        # Leave as is so that ConcatTransform can split the values
        texts = [self.eagle_processor.decode(ids, skip_special_tokens=True) for ids in data]
        return texts

    def __call__(self, data: dict) -> dict:
        return self.apply(data)
