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
import os

import torch
from torch import nn
from transformers import AutoConfig, AutoModel
from transformers.feature_extraction_utils import BatchFeature

import gr00t

DEFAULT_EAGLE_PATH = os.path.join(
    os.path.dirname(gr00t.__file__), "model", "backbone", "eagle2_hg_model"
)


class EagleBackbone(nn.Module):

    def __init__(
        self,
        tune_llm: bool = False,
        tune_visual: bool = False,
        select_layer: int = -1,
        reproject_vision: bool = False,
        use_flash_attention: bool = False,
        load_bf16: bool = False,
        eagle_path: str | None = None,
        project_to_dim: int = 1536,
    ):
        """
        Args:
            tune_llm: whether to tune the LLM model (default: True)
            tune_visual: whether to tune the visual model (default: False)
        """
        super().__init__()
        assert not reproject_vision, "Reproject vision is not implemented here, set to False"

        config = AutoConfig.from_pretrained(DEFAULT_EAGLE_PATH, trust_remote_code=True)
        self.eagle_model = AutoModel.from_config(config, trust_remote_code=True)

        if project_to_dim is not None:
            self.eagle_linear = torch.nn.Linear(2048, project_to_dim)
        else:
            self.eagle_linear = torch.nn.Identity()

        # needed since we don't use these layers. Also saves compute
        while len(self.eagle_model.language_model.model.layers) > select_layer:
            self.eagle_model.language_model.model.layers.pop(-1)

        self.select_layer = select_layer
        self.set_trainable_parameters(tune_llm, tune_visual)

        self.pattern = torch.tensor([151644, 77091, 198], device='cuda:0', dtype=torch.int64)

    def set_trainable_parameters(self, tune_llm: bool, tune_visual: bool):
        self.tune_llm = tune_llm
        self.tune_visual = tune_visual
        for p in self.parameters():
            p.requires_grad = True
        if not tune_llm:
            self.eagle_model.language_model.requires_grad_(False)
        if not tune_visual:
            self.eagle_model.vision_model.requires_grad_(False)
            self.eagle_model.mlp1.requires_grad_(False)
        print(f"Tune backbone llm: {self.tune_llm}")
        print(f"Tune backbone visual: {self.tune_visual}")
        # Check if any parameters are still trainable. If not, print a warning.
        if not tune_llm and not tune_visual:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Backbone trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No backbone trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if self.eagle_model.language_model and not self.tune_llm:
                self.eagle_model.language_model.eval()
            if self.eagle_model.vision_model and not self.tune_visual:
                self.eagle_model.vision_model.eval()

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def split_dict_by_pattern_inclusive_cuda(self, batch_dict, pattern):
        assert "input_ids" in batch_dict, "batch_dictに'input_ids'キーが必要です"
        x = batch_dict["input_ids"][0]  # shape: (N,)
        device = x.device
        pattern = pattern.to(device)

        N, M = x.shape[0], pattern.shape[0]
        if N < M:
            return batch_dict, None

        windows = x.unfold(0, M, 1)  # shape (N-M+1, M)
        matches = (windows == pattern).all(dim=1)
        idx = torch.nonzero(matches, as_tuple=False)

        if idx.numel() == 0:
            return batch_dict, None

        split_idx = idx[0].item() + M  

        a_dict, b_dict = {}, {}
        for key, val in batch_dict.items():
            a_dict[key] = val[:, :split_idx]
            b_dict[key] = val[:, split_idx:]
        return a_dict, b_dict

    def add_labels_masking_a_region(self, batch_dict, pattern, value):
        input_ids = batch_dict['input_ids']              # [B, T]
        device = input_ids.device
        pattern = pattern.to(device)
        B, T = input_ids.shape
        M = pattern.shape[0]

        labels = input_ids.clone()

        for b in range(B):
            x = input_ids[b]
            # --- パターン検索 ---
            if T < M:
                continue  # 長さがパターンより短い場合はスキップ

            # x.unfold(0, M, 1) -> [T - M + 1, M]
            windows = x.unfold(0, M, 1)
            matches = (windows == pattern).all(dim=1)  # [T - M + 1]
            idx = torch.nonzero(matches, as_tuple=False)

            # --- 該当範囲をマスク ---
            if idx.numel() > 0:
                split_idx = idx[0].item() + M  # パターン直後の位置
                labels[b, :split_idx] = value   # 例：-100でマスク

        # --- 新しい辞書返却 ---
        new_dict = {k: v for k, v in batch_dict.items()}
        new_dict['labels'] = labels
        return new_dict

    def remove_labels(self, batch_dict, pattern):
        input_ids = batch_dict["input_ids"]          # [B, T]
        device = input_ids.device
        pattern_ids = pattern.to(device)
        B, T = input_ids.shape
        M = pattern_ids.numel()

        if "attention_mask" in batch_dict:
            attn = batch_dict["attention_mask"]
        else:
            attn = torch.ones_like(input_ids, dtype=torch.long)

        new_input_ids_list = []
        new_attn_list = []
        split_idx_list = [None] * B

        for b in range(B):
            x = input_ids[b]        # [T]
            a = attn[b]             # [T]

            end = T  # デフォルト：見つからなければ全体を残す
            if T >= M:
                wins = x.unfold(0, M, 1)                 # [T-M+1, M]
                m = (wins == pattern_ids).all(dim=1)     # [T-M+1]
                idx = torch.nonzero(m, as_tuple=False)
                if idx.numel() > 0:
                    end = idx[0].item() + M              # パターン直後の位置
                    split_idx_list[b] = end

            new_input_ids_list.append(x[:end].clone())
            new_attn_list.append(a[:end].clone())

        new_dict = dict(batch_dict)  # shallow copy
        new_dict["input_ids"] = new_input_ids_list
        new_dict["attention_mask"] = new_attn_list
        return new_dict, split_idx_list


    def forward_eagle(self, vl_input: BatchFeature) -> BatchFeature:
        eagle_prefix = "eagle_"
        eagle_input = {
            k.removeprefix(eagle_prefix): v
            for k, v in vl_input.items()
            if k.startswith(eagle_prefix)
        }
        #print(eagle_input)
        #for k, v in eagle_input.items():
        #    print(k, v.shape if torch.is_tensor(v) else type(v))
        del eagle_input["image_sizes"]
        #a_dict, b_dict = self.split_dict_by_pattern_inclusive_cuda(eagle_input, self.pattern)
        #eagle_input["input_ids"] = a_dict["input_ids"]
        #eagle_input["attention_mask"] = a_dict["attention_mask"]
        #print(b_dict)
        # target_ids = b_dict.get("input_ids", None)
        # if target_ids is not None:
        #     eagle_input["labels"] = target_ids
        # print(eagle_input)
        value = -100

        #for k, v in eagle_input.items():
        #    print(k, v.shape if torch.is_tensor(v) else type(v))
        #eagle_input = self.add_labels_masking_a_region(eagle_input, self.pattern, value)
        #print(eagle_input)
        if self.training and self.tune_llm:
            eagle_input = self.add_labels_masking_a_region(eagle_input, self.pattern, value)
        #else:
        #    eagle_input, split_idx_list = self.remove_labels(eagle_input, self.pattern)
        #print(eagle_input)

        #print(eagle_input)
        #print("-----------------")
        #for k, v in eagle_input.items():
        #    print(k, v.shape if torch.is_tensor(v) else type(v))
        if self.training:
            eagle_output = self.eagle_model(**eagle_input, output_hidden_states=True, return_dict=True)
            eagle_features = eagle_output.hidden_states[self.select_layer]
        else:
            #eagle_output_org = self.eagle_model(**eagle_input, output_hidden_states=True, return_dict=True)
            eagle_output = self.eagle_model.generate(
                pixel_values=eagle_input["pixel_values"],
                input_ids=eagle_input["input_ids"],
                attention_mask=eagle_input["attention_mask"],
                max_new_tokens=64,                 # ← 系列長コントロール
                do_sample=False,                   # greedy
                temperature=1.0,
                top_p=1.0,
                #visual_features: Optional[torch.FloatTensor] = eagle_input[]
                #generation_config: Optional[GenerationConfig] = None,
                output_hidden_states=True,
                return_dict_in_generate=True)
            eagle_features = eagle_output.hidden_states[-1][self.select_layer]

        eagle_features = self.eagle_linear(eagle_features)
        if self.training and self.tune_llm:
            pred_ids = eagle_output_org.logits.argmax(dim=-1)              
            target_mask = (eagle_input['labels'] != value)                       
            pred_target_ids = [pred_ids[b][target_mask[b]] for b in range(pred_ids.size(0))]
        else:
            #print(eagle_input["input_ids"])
            lengths = eagle_input["attention_mask"].sum(dim=1)     # [B]
            #last_idx = lengths - 1
            #pred_ids = eagle_output.logits.argmax(dim=-1) 
            #print(pred_ids)
            B = eagle_input["input_ids"].size(0)
            #next_logits = eagle_output.logits[torch.arange(B, device=eagle_output.logits.device), last_idx]  # [B,V]
            #next_ids = next_logits.argmax(dim=-1, keepdim=True)
            #print(next_ids)
            
            new_texts = []
            for b in range(B):
                gen_part = eagle_output[b, :]
                new_texts.append(gen_part)
            #print(new_texts[0])
            # 3) ターゲット部分だけ抜き出し→デコード（あなたの採用方法そのまま）
            pred_target_ids = [new_texts[0]]
        
        return eagle_features, eagle_input["attention_mask"], eagle_output_org, pred_target_ids

    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()

        eagle_embeds, eagle_mask, eagle_output, pred_target_ids = self.forward_eagle(vl_input)

        # YL (TODO HACK): to resolve DDP issue when tune_visual=True
        # Ensure all trainable parameters in vision_model are used in the forward pass for DDP compatibility
        language_loss = None
        if self.training and self.tune_visual:
            dummy_term = torch.tensor(
                0.0, device=eagle_embeds.device, dtype=eagle_embeds.dtype, requires_grad=True
            )
            for param in self.eagle_model.vision_model.parameters():
                if param.requires_grad:
                    dummy_term = dummy_term + 0.0 * param.sum()
            eagle_embeds = eagle_embeds + dummy_term
            
        if self.training and self.tune_llm:
            if hasattr(eagle_output, "loss"):
                language_loss = eagle_output.loss
            elif target_ids is not None and hasattr(eagle_output, "logits"):
                # AutoModelの場合: 手動でCrossEntropy計算
                logits = eagle_output.logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = target_ids[..., 1:].contiguous()
                language_loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=self.eagle_model.config.pad_token_id,
                )
        #eagle_embeds2 = eagle_embeds.detach()

        #return BatchFeature(
        #    data={"backbone_features": eagle_embeds2, "backbone_attention_mask": eagle_mask, "backbone_lastfeature": eagle_output, "pred_target_ids": pred_target_ids, "language_loss": language_loss}
        #)  # [B, T2, hidden_size]
        return BatchFeature(
            data={"backbone_features": eagle_embeds, "backbone_attention_mask": eagle_mask, "backbone_lastfeature": eagle_output, "pred_target_ids": pred_target_ids, "language_loss": language_loss}
        )  # [B, T2, hidden_size]
