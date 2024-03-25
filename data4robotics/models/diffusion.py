# Copyright (c) Sudeep Dasari, 2023

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from data4robotics.agent import BaseAgent


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, d_model)

        Returns:
            Tensor of shape (seq_len, batch_size, d_model) with positional encodings added
        """
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class FourierFeatures(nn.Module):
    def __init__(self, time_dim, learnable=False):
        assert time_dim % 2 == 0, "time_dim must be even!"
        half_dim = int(time_dim // 2)
        super().__init__()

        w = np.log(10000) / (half_dim - 1)
        w = torch.exp(torch.arange(half_dim) * -w).float()
        self.register_parameter("w", nn.Parameter(w, requires_grad=learnable))

    def forward(self, x):
        assert len(x.shape) == 1, "assumes 1d input timestep array"
        x = x[:, None] * self.w[None]
        return torch.cat((torch.cos(x), torch.sin(x)), dim=1)


class TransformerBlock(nn.Module):
    def __init__(self, dim, cond_dim, dropout=0.1, layer_norm=True, num_heads=8):
        super().__init__()
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim)
        )
        self.map = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        self.norm_map = nn.LayerNorm(dim) if layer_norm else nn.Identity()

        self.proj_out = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(inplace=True))
        self.norm_out = nn.LayerNorm(dim) if layer_norm else nn.Identity()

    def forward(self, ac_tokens, cond_token):
        # keep re-adding the transform's global conditioning token
        cond_token = self.cond_proj(cond_token)

        tokens = torch.cat((cond_token[None], ac_tokens), 0)
        tokens, _ = self.map(tokens, tokens, tokens)
        tokens = self.norm_map(tokens[1:] + ac_tokens)
        return self.norm_out(self.proj_out(tokens) + tokens)


class NoiseNetwork(nn.Module):
    def __init__(
        self,
        ac_dim,
        global_cond_dim,
        ac_chunk,
        time_dim=256,
        hidden_dim=512,
        num_blocks=3,
        dropout=0.1,
        learnable_features=True,
    ):
        super().__init__()

        # input encoder mlps
        cond_dim = global_cond_dim + time_dim
        self.time_net = nn.Sequential(
            FourierFeatures(time_dim, learnable_features),
            nn.Linear(time_dim, time_dim),
            nn.ReLU(),
        )
        self.pos_enc = PositionalEncoding(hidden_dim, dropout=dropout)
        self.ac_proj = nn.Sequential(
            nn.Linear(ac_dim, ac_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ac_dim, hidden_dim),
        )

        # transformer blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_dim, cond_dim, dropout) for _ in range(num_blocks)]
        )

        # epsilon pred token (to prompt the transformer)
        eps_pred_tokens = torch.empty(ac_chunk, 1, hidden_dim)
        nn.init.xavier_uniform_(eps_pred_tokens)
        eps_pred_tokens = nn.Parameter(eps_pred_tokens, requires_grad=True)
        self.register_parameter("eps_pred_tokens", nn.Parameter(eps_pred_tokens))

        # turns predicted tokens into epsilons
        self.eps_out = nn.Linear(hidden_dim, ac_dim)

        print(
            "number of diffusion parameters: {:e}".format(
                sum(p.numel() for p in self.parameters())
            )
        )

    def forward(self, noise_actions, time, obs_enc):
        B, T = noise_actions.shape[:2]
        time_enc = self.time_net(time)
        cond_token = torch.cat((obs_enc, time_enc), -1)

        ac_tokens = self.ac_proj(noise_actions)
        ac_tokens = ac_tokens.transpose(0, 1)
        ac_tokens = torch.cat((ac_tokens, self.eps_pred_tokens.repeat((1, B, 1))), 0)
        ac_tokens = self.pos_enc(ac_tokens)

        # apply transformer blocks and transpose back
        for block in self.blocks:
            ac_tokens = block(ac_tokens, cond_token)
        ac_tokens = ac_tokens[-T:].transpose(0, 1)

        # apply final epsilon prediction layer
        return self.eps_out(ac_tokens)


class DiffusionTransformerAgent(BaseAgent):
    def __init__(
        self,
        features,
        odim,
        n_cams,
        use_obs,
        ac_dim,
        ac_chunk,
        train_diffusion_steps,
        eval_diffusion_steps,
        imgs_per_cam=1,
        dropout=0,
        share_cam_features=False,
        feat_norm=None,
        token_dim=None,
        noise_net_kwargs=dict(),
    ):

        # initialize obs and img tokenizers
        super().__init__(
            odim=odim,
            features=features,
            n_cams=n_cams,
            imgs_per_cam=imgs_per_cam,
            use_obs=use_obs,
            share_cam_features=share_cam_features,
            dropout=dropout,
            feat_norm=feat_norm,
            token_dim=token_dim,
        )

        cond_dim = self.n_tokens * self.token_dim
        self.noise_net = NoiseNetwork(
            ac_dim=ac_dim,
            ac_chunk=ac_chunk,
            global_cond_dim=cond_dim,
            **noise_net_kwargs,
        )
        self._ac_dim, self._ac_chunk = ac_dim, ac_chunk

        assert (
            eval_diffusion_steps <= train_diffusion_steps
        ), "Can't eval with more steps!"
        self._train_diffusion_steps = train_diffusion_steps
        self._eval_diffusion_steps = eval_diffusion_steps
        self.diffusion_schedule = DDIMScheduler(
            num_train_timesteps=train_diffusion_steps,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type="epsilon",
        )

    def forward(self, imgs, obs, ac_flat, mask_flat):
        # get observation encoding and sample noise/timesteps
        B, device = obs.shape[0], obs.device
        s_t = self.tokenize_obs(imgs, obs, flatten=True)
        timesteps = torch.randint(
            low=0, high=self._train_diffusion_steps, size=(B,), device=device
        ).long()

        # diffusion unet logic assumes [B, T, adim]
        mask = mask_flat.reshape((B, self.ac_chunk, self.ac_dim))
        actions = ac_flat.reshape((B, self.ac_chunk, self.ac_dim))
        noise = torch.randn_like(actions)

        # construct noise actions given real actions, noise, and diffusion schedule
        noise_acs = self.diffusion_schedule.add_noise(actions, noise, timesteps)
        noise_pred = self.noise_net(noise_acs, timesteps, s_t)

        # calculate loss for noise net
        loss = nn.functional.mse_loss(noise_pred, noise, reduction="none")
        loss = (loss * mask).sum(1)  # mask the loss to only consider "real" acs
        return loss.mean()

    def get_actions(self, imgs, obs, n_steps=None):
        # get observation encoding and sample noise
        B, device = obs.shape[0], obs.device
        s_t = self.tokenize_obs(imgs, obs, flatten=True)
        noise_actions = torch.randn(B, self.ac_chunk, self.ac_dim, device=device)

        # set number of steps
        eval_steps = self._eval_diffusion_steps
        if n_steps is not None:
            assert (
                n_steps <= self._train_diffusion_steps
            ), f"can't be > {self._train_diffusion_steps}"
            eval_steps = n_steps

        # begin diffusion process
        self.diffusion_schedule.set_timesteps(eval_steps)
        self.diffusion_schedule.alphas_cumprod = (
            self.diffusion_schedule.alphas_cumprod.to(device)
        )
        for timestep in self.diffusion_schedule.timesteps:
            # predict noise given timestep
            batched_timestep = timestep.unsqueeze(0).repeat(B).to(device)
            noise_pred = self.noise_net(noise_actions, batched_timestep, s_t)

            # take diffusion step
            noise_actions = self.diffusion_schedule.step(
                model_output=noise_pred, timestep=timestep, sample=noise_actions
            ).prev_sample

        # return final action post diffusion
        return noise_actions

    @property
    def ac_chunk(self):
        return self._ac_chunk

    @property
    def ac_dim(self):
        return self._ac_dim
