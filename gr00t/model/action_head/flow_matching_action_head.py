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

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

from gr00t.model.action_head.action_encoder import (
    SinusoidalPositionalEncoding,
    swish,
)

from .cross_attention_dit import DiT, SelfAttentionTransformer
import math


class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        # For each category, we have separate weights and biases.
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    # def forward(self, x, cat_ids):
    #     selected_W = self.W[cat_ids]
    #     selected_b = self.b[cat_ids]
    #     return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)

    def forward(self, x, cat_ids):
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        # ★ 追加: 入力 x の dtype に一時キャスト（勾配は A にだけ必要なので OK）
        if selected_W.dtype != x.dtype:
            selected_W = selected_W.to(x.dtype)
            selected_b = selected_b.to(x.dtype)
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)


class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size, num_embodiments):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)  # (d -> w)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)  # (2w -> w)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)  # (w -> w)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        cat_ids:   shape (B,)
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps
        #    so that shape => (B, T)
        #    e.g. if timesteps is (B,), replicate across T
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError(
                "Expected `timesteps` to have shape (B,) so we can replicate across T."
            )

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.W1(actions, cat_ids)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then W2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))

        # 5) Finally W3 => (B, T, w)
        x = self.W3(x, cat_ids)
        return x


@dataclass
class FlowmatchingActionHeadConfig(PretrainedConfig):
    """NOTE: N1.5 uses XEmbFlowmatchingPolicyHeadConfig as action head"""

    add_pos_embed: bool = field(
        default=True, metadata={"help": "Whether to add positional embedding"}
    )
    model_dtype: str = field(default="float32", metadata={"help": "Model data type."})
    diffusion_model_cfg: dict = field(
        default=None, metadata={"help": "Diffusion model configuration."}
    )
    input_embedding_dim: int = field(
        default=1536, metadata={"help": "Input embedding channel dimension."}
    )
    backbone_embedding_dim: int = field(
        default=1536, metadata={"help": "Backbone embedding channel dimension."}
    )

    hidden_size: int = field(default=1024, metadata={"help": "Input embedding dimension."})
    max_seq_len: int = field(default=1024, metadata={"help": "Maxium Sequence Length"})
    action_dim: int = field(default=None, metadata={"help": "Action dimension."})
    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})
    noise_beta_alpha: float = field(default=1.5, metadata={"help": ""})
    noise_beta_beta: float = field(default=1.0, metadata={"help": ""})
    noise_s: float = field(
        default=0.999, metadata={"help": "Flow matching noise Beta distribution s."}
    )
    num_timestep_buckets: int = field(
        default=1000, metadata={"help": "Number of timestep discretization buckets."}
    )
    num_inference_timesteps: int = field(
        default=None,
        metadata={"help": "Number of inference steps for noise diffusion."},
    )
    max_num_embodiments: int = field(default=32, metadata={"help": "Number of embodiments."})
    tune_projector: bool = field(default=True, metadata={"help": "Whether to tune the projector."})
    tune_diffusion_model: bool = field(
        default=True, metadata={"help": "Whether to tune the diffusion model."}
    )
    load_pretrained_det_decode_layer_path: str = field(
        default=None, metadata={"help": "Path to pretrained detection model."}
    )
    detection_coeff: float = field(default=1.0, metadata={"help": "Detection coefficient."})

    freeze_decode_layer: bool = field(default=False)
    expand_batch: int = field(default=None)
    use_vlln: bool = field(default=True)

    vl_self_attention_cfg: dict = field(default=None)
    num_target_vision_tokens: int = field(
        default=32, metadata={"help": "Number of target vision tokens."}
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class FlowmatchingActionHead(nn.Module):
    config_class = FlowmatchingActionHeadConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: FlowmatchingActionHeadConfig,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        self.model = DiT(**config.diffusion_model_cfg)
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=config.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )
        self.future_tokens = nn.Embedding(config.num_target_vision_tokens, self.input_embedding_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

        self.vlln = (
            nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        )
        self.vl_self_attention = (
            SelfAttentionTransformer(**config.vl_self_attention_cfg)
            if config.use_vlln
            else nn.Identity()
        )

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        self.config = config
        self.set_trainable_parameters(config.tune_projector, config.tune_diffusion_model)

    def set_trainable_parameters(self, tune_projector: bool, tune_diffusion_model: bool):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        for p in self.parameters():
            p.requires_grad = True
        if not tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        print(f"Tune action head projector: {self.tune_projector}")
        print(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        # Check if any parameters are still trainable. If not, print a warning.
        if not tune_projector and not tune_diffusion_model:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Action head trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No action head trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.config.noise_s - sample) / self.config.noise_s

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        backbone_features = backbone_output["backbone_features"]
        backbone_features = self.vlln(backbone_features)
        backbone_features = self.vl_self_attention(backbone_features)
        backbone_output["backbone_features"] = backbone_features
        return backbone_output

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        backbone_output = self.process_backbone_output(backbone_output)

        if self.config.expand_batch is not None:
            for k, v in backbone_output.items():
                ndim = len(v.shape)
                factors = [self.config.expand_batch]
                while len(factors) < ndim:
                    factors.append(1)
                factors = tuple(factors)
                expanded = v.repeat(*factors)
                backbone_output[k] = expanded

            for k, v in action_input.items():
                ndim = len(v.shape)
                factors = [self.config.expand_batch]
                while len(factors) < ndim:
                    factors.append(1)
                factors = tuple(factors)
                expanded = v.repeat(*factors)
                action_input[k] = expanded

        # Get vision and language embeddings.
        vl_embs = backbone_output.backbone_features
        device = vl_embs.device

        # Get embodiment ID.
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Embed noised action trajectory.
        actions = action_input.action
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # Join vision, language, state and action embedding along sequence dimension.
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
        sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

        vl_attn_mask = backbone_output.backbone_attention_mask

        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=vl_attn_mask,
            timestep=t_discretized,
            return_all_hidden_states=False,  # NOTE (YL): not using flare now
        )
        pred = self.action_decoder(model_output, embodiment_id)
        pred_actions = pred[:, -actions.shape[1] :]

        # Slice out only the action portion of pred and target.
        action_mask = action_input.action_mask
        loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
        loss = loss.sum() / action_mask.sum()
        output_dict = {
            "loss": loss,
        }
        return BatchFeature(data=output_dict)

    @torch.no_grad()
    def get_action(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:

        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embs = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Set initial actions as the sampled noise.
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.config.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        # Run denoising steps.
        for t in range(num_steps):
            t_cont = t / float(num_steps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            

            # Embed noised action trajectory.
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
            # Maybe add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            # Join vision, language, state and action embedding along sequence dimension.
            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
            sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

            # Run model forward.
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                timestep=timesteps_tensor,
            )
            pred = self.action_decoder(model_output, embodiment_id)

            pred_velocity = pred[:, -self.action_horizon :]

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity

        return BatchFeature(data={"action_pred": actions})

    def get_action_rtc(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:

        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embs = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Set initial actions as the sampled noise.
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.config.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        def _finite_or_zero(x: torch.Tensor, name: str):
            # 有限かどうかのマスク
            mask = torch.isfinite(x)
            bad = (~mask).sum().item()

            if bad > 0:
                # 有限値だけで min/max を計算（全部 bad のときは NaN を出す）
                if mask.any():
                    x_min = x[mask].min().item()
                    x_max = x[mask].max().item()
                else:
                    x_min = float("nan")
                    x_max = float("nan")

                print(f"[RTC] non-finite in {name}: "
                    f"min={x_min:.3e} max={x_max:.3e} n_bad={bad}")

                # NaN/±Inf を安全な値に置換して返す
                x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
            return x

        def _norm(x):
            return torch.linalg.vector_norm(x.reshape(x.shape[0], -1), dim=1, keepdim=True) + 1e-8

        def summarize(x, name="x"):
            finite = torch.isfinite(x)
            n_all  = x.numel()
            n_bad  = (~finite).sum().item()
            print(f"[{name}] shape={tuple(x.shape)} dtype={x.dtype} "
                f"finite={(n_all - n_bad)}/{n_all}  bad={n_bad}")
            if n_bad:
                # どの軸に NaN/Inf が出てるか（B,H,D の想定）
                bad_H = (~finite).any(dim=-1).any(dim=0)  # H 方向
                bad_D = (~finite).any(dim=1).any(dim=0)   # D 方向
                print(f"  bad H idx: {bad_H.nonzero(as_tuple=True)[0].tolist()}")
                print(f"  bad D idx: {bad_D.nonzero(as_tuple=True)[0].tolist()}")
            else:
                print(f"  min={x.min().item():.3e} max={x.max().item():.3e} mean={x.mean().item():.3e}")


        # Run denoising steps.
        for t in range(num_steps):
            t_cont = t / float(num_steps)                    # τ in [0,1)
            t_discretized = int(t_cont * self.num_timestep_buckets)
            dt = 1.0 / num_steps

            # --- Optional: RTC ΠGDM guidance inputs ---
            try:
                Y = action_input["rtc_prev_action"]          # [B,H,D] (normalized space)
                W = action_input["rtc_weight_mask"]          # [B,H,D] or [B,H,1]
                beta = action_input.get("rtc_beta", 0.0)
                guidance_clip = action_input.get("rtc_guidance_clip", 1.0)
                angle_idx = action_input.get("rtc_angle_indices", None)
                use_guidance = True
            except KeyError:
                Y = None; W = None; beta = 0.0; guidance_clip = 1.0; angle_idx = None
                use_guidance = False

            if t_cont < 0.15: 
                use_guidance = False

            # すべてのテンソルを同じ device/dtype に揃える
            if use_guidance:
                if W.dim() == 3 and W.shape[-1] == 1:
                    W = W.expand(-1, -1, self.action_dim)    # [B,H,1] -> [B,H,D]
                Y = Y.to(device=vl_embs.device, dtype=vl_embs.dtype)
                W = W.to(device=vl_embs.device, dtype=vl_embs.dtype)
                if not torch.is_tensor(beta): beta = torch.tensor(float(beta), device=vl_embs.device, dtype=vl_embs.dtype)
                if not torch.is_tensor(guidance_clip): guidance_clip = torch.tensor(float(guidance_clip), device=vl_embs.device, dtype=vl_embs.dtype)

            K_warmup = max(2, num_steps // 16)   # 例: 32stepなら前半2ステップは無効
            rtc_enabled_this_step = use_guidance and (t >= K_warmup) and (beta > 0)

            # ---- enable grad only for this iteration (ΠGDM needs VJP wrt actions) ----
            with torch.enable_grad():
                A = actions.detach().requires_grad_(True)     # track grad only wrt A
                #A = actions.detach().to(torch.float32).requires_grad_(True)
                A32 = actions.detach().to(torch.float32).requires_grad_(True)

                # state_features32 = state_features.to(torch.float32)
                # vl_embs32 = vl_embs.to(torch.float32)

                # 1) エンコード（actions=A を使う）
               
                timesteps_tensor = torch.full(size=(batch_size,), fill_value=t_discretized, device=device)
                action_features = self.action_encoder(A, timesteps_tensor, embodiment_id)
                if self.config.add_pos_embed:
                    pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                    pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                    #pos_embs = self.position_embedding(pos_ids).unsqueeze(0).to(torch.float32)
                    action_features = action_features + pos_embs

                # 2) ビジョン＋状態と結合
                future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
                sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)
                # future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs32.shape[0], -1, -1).to(torch.float32)
                # sa_embs = torch.cat((state_features32, future_tokens, action_features), dim=1)

                # 3) モデル前向き → 速度予測 v
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embs,
                    timestep=timesteps_tensor,
                )
                pred = self.action_decoder(model_output, embodiment_id)

                # model_output = self.model(
                #     hidden_states=sa_embs,
                #     encoder_hidden_states=vl_embs32,
                #     timestep=timesteps_tensor,
                # )
                # pred = self.action_decoder(model_output, embodiment_id).to(torch.float32)

                v = pred[:, -self.action_horizon :]           # [B,H,D] = pred_velocity

                v = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
                v_clip = 3.0                      # 正規化空間なので 2〜4 程度が無難
                v = torch.clamp(v, -v_clip, v_clip)

                if not torch.isfinite(v).all():
                    print(f"[RTC] v non-finite at t={t}. zero-step update.")
                    # A は更新しない（= その反復はスキップ）
                    actions = A16.detach()        # 次反復へ
                    continue

                if not torch.isfinite(v).all():
                    # 何が壊れているかをログ
                    bad = (~torch.isfinite(v)).sum().item()
                    print(f"[RTC] v has non-finite ({bad} elems). Skip guidance at t={t}.")
                    use_guidance_step = False
                else:
                    use_guidance_step = use_guidance and (t >= K_warmup)  # 例: warmup_steps = 5

                # v32 = v.to(torch.float32)
                # v32 = _finite_or_zero(v32, "v32")
                rtc_enabled = (use_guidance and (t >= K_warmup) and (beta > 0.0))

                A_hat1 = A + (1.0 - t_cont) * v
                # summarize(v,       "v")
                # summarize(A_hat1,  "A_hat1")

                if use_guidance and beta.item() > 0.0 and rtc_enabled:
                    # 4) 1歩先のデノイズ写像 Â1 = A + (1-τ)*v

                    dim_mask = action_input.get("rtc_dim_mask", None)  # [1,1,D] (0/1)
                    if dim_mask is not None:
                        # bf16/ptr の都合があるので dtype はA32に合わせる
                        dim_mask32 = dim_mask.to(A32.device, dtype=A32.dtype)
                        # A32 のパディングは更新しない（勾配も流れるが、後段で e/g にも同じマスクを掛ける）
                        A32 = A32 * dim_mask32

                    A_hat1_32 = A_hat1.to(torch.float32)
                    A_hat1_32 = _finite_or_zero(A_hat1, "A_hat1")

                    Y32 = Y.to(torch.float32)
                    #W32 = W.to(torch.float32)
                    W32 = (W if W.dim() == 3 else W.unsqueeze(-1).expand(-1, -1, self.action_dim)).to(torch.float32)

                    if dim_mask is not None:
                        W32 = W32 * dim_mask.to(W32.dtype)

                    # 5) 誤差 e = (Y - Â1) ⊙ W  （角度DoFはwrap-to-π）
                    e = (Y32 - A_hat1_32)
                    if angle_idx is not None and len(angle_idx) > 0:
                        # wrap-to-pi for specified dims
                        # e[..., i] = atan2(sin(e[..., i]), cos(e[..., i]))
                        idx = torch.as_tensor(angle_idx, device=e.device, dtype=torch.long)
                        e_ang = torch.atan2(torch.sin(e.index_select(-1, idx)),
                                            torch.cos(e.index_select(-1, idx)))
                        e = e.scatter(-1, idx.unsqueeze(0).unsqueeze(0).expand(e.shape[0], e.shape[1], -1), e_ang)

                    e_clip = 0.05                # まずは 0.25〜0.5 程度
                    e = torch.clamp(e, -e_clip, e_clip)
                    e = e * W32
                    e = _finite_or_zero(e, "e")

                    e16 = e.to(A_hat1.dtype)

                    try:
                        g = torch.autograd.grad(
                            outputs=A_hat1, inputs=A, grad_outputs=e16,
                            retain_graph=False, create_graph=False, allow_unused=False
                        )[0]
                    except Exception as ex:
                        print(f"[RTC] grad failed at t={t}: {ex}. Skip guidance this step.")
                        g = None

                    g = _finite_or_zero(g, "g")

                    if g is None or not torch.isfinite(g).all():
                        print(f"[RTC] non-finite grad at t={t}. Skip guidance this step.")
                        A_next = A + dt * v
                    else:
                        g32 = g.to(torch.float32)
                        g32 = torch.nan_to_num(g32, nan=0.0, posinf=1e3, neginf=-1e3)
                        g_norm = torch.linalg.norm(g32.reshape(g32.shape[0], -1), dim=1, keepdim=True) + 1e-8
                        g_max = 0.1    # まずは 1.0 前後
                        scale = torch.clamp(g_max / g_norm, max=1.0).view(-1, 1, 1)
                        g32 = g32 * scale

                        # rel = (_norm(v32) / _norm(g)).clamp(max=2.0).view(-1, 1, 1)
                        # g = g * rel



                        # 7) 係数 α(τ) = min(β, (1-τ)/(τ * r_τ^2)) （数値安定のためεつき）
                        # tau = torch.tensor(max(t_cont, 1e-4), device=device, dtype=vl_embs.dtype)
                        # r2 = ((1.0 - tau) ** 2) / (tau ** 2 + (1.0 - tau) ** 2)
                        # alpha = torch.minimum(beta, (1.0 - tau) / (tau * r2))
                        # alpha = beta * (1.0 - t_cont)
                        # alpha = min(alpha, 0.5)

                        # VJP 区間は FP32 前提
                        dtype_vjp = torch.float32
                        device = g32.device        # A32: いま勾配を流しているアクション（[B,H,D]）

                        eps = 1e-6

                        # t_cont を必ず Tensor[FP32] に
                        # 例: t_cont はスカラーでもバッチでもOK（[B] を想定）
                        if not torch.is_tensor(t_cont):
                            t_cont = torch.tensor(t_cont, device=device, dtype=dtype_vjp)
                        else:
                            t_cont = t_cont.to(device=device, dtype=dtype_vjp)

                        # clamp は Tensor 入力で
                        tau = torch.clamp(t_cont, min=eps, max=1.0 - eps)

                        # r_τ^2 と α の論文式（数値安定: 分母に eps）
                        r2 = ((1.0 - tau) ** 2) / (tau**2 + (1.0 - tau)**2 + eps)
                        alpha_raw = (1.0 - tau) / (tau * r2 + eps)

                        # beta も Tensor[FP32] に統一
                        if not torch.is_tensor(beta):
                            beta = torch.tensor(float(beta), device=device, dtype=dtype_vjp)
                        else:
                            beta = beta.to(device=device, dtype=dtype_vjp)

                        alpha = torch.minimum(beta, alpha_raw)  # shape: scalar or [B]

                        # v/g へのブロードキャストに備えて [B,1,1] へ整形
                        B = A32.shape[0]
                        if alpha.dim() == 0:
                            alpha = alpha.view(1,1,1).expand(B,1,1)
                        elif alpha.dim() == 1:
                            alpha = alpha[:, None, None]        # [B] -> [B,1,1]




                        # 8) 速度に加算して更新
                        v_guided = v.to(torch.float32) + alpha * g32

                        # delta = dt * (v + alpha * g) - dt * v   # = dt * alpha * g
                        # r = 0.05  # まずは 0.03〜0.1
                        # delta = delta * (r / (delta.reshape(B,-1).norm(dim=1, keepdim=True)+1e-8)).clamp(max=1.0).view(B,H,D)
                        # A_next = A32 + dt * v32 + delta

                        A_next = A + dt * v_guided.to(v.dtype)


                else:
                    # ガイダンス無し
                    A_next = A + dt * v
                
                # a_min = -3.0
                # a_max =  3.0
                # A_next = torch.clamp(A_next, a_min, a_max)



            # 9) 先頭 d ステップのハード凍結（W≈1のところをYで上書き）
            if use_guidance and rtc_enabled:
                freeze_mask = (W >= 0.999)
                A_next = torch.where(freeze_mask, Y, A_next)

            #actions = A_next.detach()  # 次反復へ
            actions = A_next.to(vl_embs.dtype).detach()

        return BatchFeature(data={"action_pred": actions})


    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
