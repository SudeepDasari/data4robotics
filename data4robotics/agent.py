# Copyright (c) Sudeep Dasari, 2023

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
from torch import nn


class Agent(nn.Module):
    def __init__(self, features, policy, shared_mlp, odim, 
                       n_cams, use_obs, dropout=0):
        super().__init__()

        # store visual, policy, and inverse model
        self.visual_features = features
        self._policy = policy

        # build shared mlp layers
        self._odim = odim if use_obs else 0
        self._use_obs, self._n_cams = bool(use_obs), n_cams
        mlp_in = self._odim + n_cams * features.embed_dim
        mlp_def = [mlp_in] + shared_mlp
        layers = [nn.BatchNorm1d(num_features=mlp_in)]
        for i, o in zip(mlp_def[:-1], mlp_def[1:]):
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(i, o))
            layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        self._shared_mlp = nn.Sequential(*layers)

    def forward(self, imgs, obs, zero_std=False):
        s_t = self._shared_forward(imgs, obs)
        action_dist = self._policy(s_t, zero_std=zero_std)
        return action_dist

    def get_actions(self, img, obs, zero_std=True):
        policy_in = self._shared_forward(img, obs)
        return self._policy.get_actions(policy_in, zero_std=zero_std)
    
    def _shared_forward(self, imgs, obs):
        shared_in = torch.cat((self.embed(imgs), obs), dim=1) if self._use_obs \
                    else self.embed(imgs)
        return self._shared_mlp(shared_in)
    
    def embed(self, imgs):
        if len(imgs.shape) == 5:
            B, N, C, H, W = imgs.shape
            embeds = self.visual_features(imgs.reshape((B * N, C, H, W)))
            embeds = embeds.reshape((B, N * self.visual_features.embed_dim))
            return embeds
        return self.visual_features(imgs)

    @property
    def odim(self):
        return self._odim
    
    @property
    def n_cams(self):
        return self._n_cams
    
    @property
    def ac_chunk(self):
        return self._policy.ac_chunk

    def restore_features(self, restore_path):
        if not restore_path:
            print('No restore path supplied!')
            return
        state_dict = torch.load(restore_path, map_location='cpu')['features']
        self.visual_features.load_state_dict(state_dict)
        print(f"Restored {restore_path}!")
