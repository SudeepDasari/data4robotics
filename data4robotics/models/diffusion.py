# Copyright (c) Sudeep Dasari, 2023

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import numpy as np
import torch.nn as nn
from data4robotics.agent import Agent
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


class FourierFeatures(nn.Module):
    def __init__(self, time_dim, learnable=False):
        assert time_dim % 2 == 0, "time_dim must be even!"
        half_dim = int(time_dim // 2)
        super().__init__()

        w = np.log(10000) / (half_dim - 1)
        w = torch.exp(torch.arange(half_dim) * -w).float()
        self.register_parameter('w', nn.Parameter(w, requires_grad=learnable))
    
    def forward(self, x):
        assert len(x.shape) == 1, "assumes 1d input timestep array"
        x = x[:,None] * self.w[None]
        return torch.cat((torch.cos(x), torch.sin(x)), dim=1)


class MLPResNetBlock(nn.Module):
    def __init__(self, dim, dropout=0.1, layer_norm=True, alpha=2):
        super().__init__()

        layers = [nn.Dropout(dropout)]
        if layer_norm:
            layers.append(nn.LayerNorm(dim))
        
        layers.append(nn.Linear(dim, alpha * dim))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(alpha * dim, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.net(x)


class NoiseNetwork(nn.Module):
    def __init__(self, adim, ac_chunk, cond_dim, time_dim=32, learnable_features=True,
                  num_blocks=3, hidden_dim=256, dropout_rate=0.1, 
                  use_layer_norm=True):
        super().__init__()

        self.time_net = nn.Sequential(FourierFeatures(time_dim, learnable_features),
                                      nn.Linear(time_dim, time_dim), nn.ReLU())
        in_dim = adim * ac_chunk + cond_dim + time_dim
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU())
        net = [MLPResNetBlock(hidden_dim, dropout_rate, use_layer_norm)
                                                for _ in range(num_blocks)]
        self.net = nn.Sequential(*net)
        self.out = nn.Sequential(nn.ReLU(), nn.Linear(hidden_dim, adim * ac_chunk))
    
    def forward(self, obs_enc, noise_ac_flat, time):
        time_enc = self.time_net(time)
        x = self.proj(torch.cat((noise_ac_flat, obs_enc, time_enc), -1))
        x = self.net(x)
        return self.out(x)


class DiffusionAgent(Agent):
    def __init__(self, features, shared_mlp, odim, n_cams, use_obs, 
                 ac_dim, ac_chunk, diffusion_steps, dropout=0,
                 noise_net_kwargs=dict()):
        super().__init__(features, None, shared_mlp, odim, n_cams, 
                         use_obs, dropout)

        self.noise_net = NoiseNetwork(adim=ac_dim, ac_chunk=ac_chunk, 
                                      cond_dim=self.obs_enc_dim,
                                      **noise_net_kwargs)
        
        self._ac_dim, self._ac_chunk = ac_dim, ac_chunk
        self._diffusion_steps = diffusion_steps
        self.diffusion_schedule = DDPMScheduler(
                                    num_train_timesteps=diffusion_steps,
                                    beta_schedule="squaredcos_cap_v2",
                                    clip_sample=True,
                                    prediction_type="epsilon"
                                    )
        
    def forward(self, imgs, obs, flat_actions):
        # get observation encoding and sample noise/timesteps
        B, device = imgs.shape[0], imgs.device
        s_t = self._shared_forward(imgs, obs)
        noise = torch.randn_like(flat_actions)
        timesteps = torch.randint(low=0, high=self._diffusion_steps, size=(B,), 
                                  device=device).long()

        # construct noise actions given real actions, noise, and diffusion schedule
        noise_acs = self.diffusion_schedule.add_noise(flat_actions, noise, timesteps)
        noise_pred = self.noise_net(s_t, noise_acs, timesteps)
        
        # calculate loss for noise net
        loss = nn.functional.mse_loss(noise_pred, noise, reduction="none").sum(dim=1)
        loss = loss.mean()
        return loss

    def get_actions(self, imgs, obs):
        # get observation encoding and sample noise
        B, device = imgs.shape[0], imgs.device
        s_t = self._shared_forward(imgs, obs)
        noise_actions = torch.randn(B, self.ac_dim * self.ac_chunk, device=device)

        # begin diffusion process
        self.diffusion_schedule.set_timesteps(self._diffusion_steps)
        self.diffusion_schedule.alphas_cumprod = self.diffusion_schedule.alphas_cumprod.to(device)
        for timestep in self.diffusion_schedule.timesteps:
            # predict noise given timestep
            time = timestep.unsqueeze(0).repeat(B).to(device)
            noise_pred = self.noise_net(s_t, noise_actions, time)

            # take diffusion step
            noise_actions = self.diffusion_schedule.step(model_output=noise_pred, 
                                                timestep=timestep, 
                                                sample=noise_actions).prev_sample

        # return final action post diffusion
        return noise_actions.reshape((B, self.ac_chunk, self.ac_dim))
    
    @property
    def ac_chunk(self):
        return self._ac_chunk

    @property
    def ac_dim(self):
        return self._ac_dim
