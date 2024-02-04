# Copyright (c) Sudeep Dasari, 2023

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import numpy as np
import torch.nn as nn
from data4robotics.agent import Agent
from diffusers.schedulers.scheduling_ddim import DDIMScheduler


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
        layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.net(x)


class MLPResFILMBlock(nn.Module):
    def __init__(self, dim, cond_dim, dropout=0.1, layer_norm=True, alpha=2):
        super().__init__()

        top_layers = [nn.Dropout(dropout)]
        if layer_norm:
            top_layers.append(nn.LayerNorm(dim))
        
        top_layers.append(nn.Linear(dim, alpha * dim))
        top_layers.append(nn.ReLU())
        self.top_layers = nn.Sequential(*top_layers)

        self.film_embed = nn.Sequential(nn.Linear(cond_dim, alpha * dim),
                                        nn.ReLU())
        self.out_layers = nn.Sequential(nn.Linear(alpha * dim, dim),
                                        nn.ReLU())                               

    def forward(self, x, cond):
        out = self.top_layers(x)
        out = self.film_embed(cond) + out
        out = self.out_layers(out)
        return x + out


class NoiseNetwork(nn.Module):
    def __init__(self, adim, ac_chunk, cond_dim, time_dim=32, learnable_features=True,
                  num_blocks=3, hidden_dim=256, dropout=0.1, 
                  use_layer_norm=True):
        super().__init__()

        self.time_net = nn.Sequential(FourierFeatures(time_dim, learnable_features),
                                      nn.Linear(time_dim, time_dim), nn.ReLU())
        in_dim = adim * ac_chunk + cond_dim + time_dim
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU())
        net = [MLPResFILMBlock(hidden_dim, in_dim, dropout, use_layer_norm)
                                                      for _ in range(num_blocks)]
        self.blocks = nn.ModuleList(net)
        self.out = nn.Linear(hidden_dim, adim * ac_chunk)

        print("number of diffusion parameters: {:e}".format(sum(p.numel() for p in self.parameters())))
    
    def forward(self, noise_ac_flat, time, obs_enc):
        time_enc = self.time_net(time)
        cond = torch.cat((noise_ac_flat, obs_enc, time_enc), -1)
        
        # apply diffusion mlp blocks
        x = self.proj(cond)
        for b in self.blocks:
            x = b(x, cond)
        
        # apply final linear layer
        return self.out(x)


class DiffusionMLPAgent(Agent):
    def __init__(self, features, shared_mlp, odim, n_cams, use_obs, 
                 ac_dim, ac_chunk, train_diffusion_steps, eval_diffusion_steps,
                 imgs_per_cam=1, dropout=0, share_cam_features=False, 
                 feat_batch_norm=True, noise_net_kwargs=dict()):
        super().__init__(features, None, shared_mlp, odim, n_cams, 
                         use_obs, imgs_per_cam, dropout, share_cam_features,
                         feat_batch_norm)
        self.noise_net = NoiseNetwork(adim=ac_dim, ac_chunk=ac_chunk, 
                                      cond_dim=self.obs_enc_dim,
                                      **noise_net_kwargs)
        self._ac_dim, self._ac_chunk = ac_dim, ac_chunk
        
        assert eval_diffusion_steps <= train_diffusion_steps, "Can't eval with more steps!"
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
                                    prediction_type="epsilon"
                                    )
        
    def forward(self, imgs, obs, actions, mask):
        # get observation encoding and sample noise/timesteps
        B, device = obs.shape[0], obs.device
        s_t = self._shared_forward(imgs, obs)
        timesteps = torch.randint(low=0, high=self._train_diffusion_steps, size=(B,), 
                                  device=device).long()
        noise = torch.randn_like(actions)

        # construct noise actions given real actions, noise, and diffusion schedule
        noise_acs = self.diffusion_schedule.add_noise(actions, noise, timesteps)
        noise_pred = self.noise_net(noise_acs, timesteps, s_t)
        
        # calculate loss for noise net
        loss = nn.functional.mse_loss(noise_pred, noise, reduction="none")
        loss = (loss * mask).sum(1)    # mask the loss to only consider "real" acs
        return loss.mean()

    def get_actions(self, imgs, obs, n_steps=None):
        # get observation encoding and sample noise
        B, device = obs.shape[0], obs.device
        s_t = self._shared_forward(imgs, obs)
        noise_actions = torch.randn(B, self.ac_chunk * self.ac_dim, device=device)

        # set number of steps
        eval_steps = self._eval_diffusion_steps
        if n_steps is not None:
            assert n_steps <= self._train_diffusion_steps, \
                  f"can't be > {self._train_diffusion_steps}"
            eval_steps = n_steps
        
        # begin diffusion process
        self.diffusion_schedule.set_timesteps(eval_steps)
        self.diffusion_schedule.alphas_cumprod = self.diffusion_schedule.alphas_cumprod.to(device)
        for timestep in self.diffusion_schedule.timesteps:
            # predict noise given timestep
            batched_timestep = timestep.unsqueeze(0).repeat(B).to(device)
            noise_pred = self.noise_net(noise_actions, batched_timestep, s_t)

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
