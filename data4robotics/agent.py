# Copyright (c) Sudeep Dasari, 2023

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch, copy
from torch import nn


class Agent(nn.Module):
    def __init__(self, features, policy, shared_mlp, odim, 
                       n_cams, use_obs, imgs_per_cam=1, dropout=0, 
                       share_cam_features=False, feat_batch_norm=True):
        super().__init__()

        # store visual features (duplicate weights if shared)
        self._share_cam_features = share_cam_features
        self.embed_dim = features.embed_dim * n_cams * imgs_per_cam
        if self._share_cam_features:
            self.visual_features = features
        else:
            feat_list = [features] + [copy.deepcopy(features) \
                                      for _ in range(1, n_cams)]
            self.visual_features = nn.ModuleList(feat_list)
        
        # store policy network
        self._policy = policy

        # build shared mlp layers
        self._odim = odim if use_obs else 0
        self._use_obs, self._n_cams = bool(use_obs), n_cams
        mlp_in = self._odim + self.embed_dim
        mlp_def = [mlp_in] + shared_mlp
        layers = [nn.BatchNorm1d(num_features=mlp_in)] if feat_batch_norm \
                 else []
        for i, o in zip(mlp_def[:-1], mlp_def[1:]):
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(i, o))
            layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        self._shared_mlp = nn.Sequential(*layers)
        self.obs_enc_dim = mlp_def[-1]

    def forward(self, imgs, obs, ac_flat, mask_flat):
        s_t = self._shared_forward(imgs, obs)
        action_dist = self._policy(s_t)
        loss = -torch.mean(action_dist.masked_log_prob(ac_flat, mask_flat)) \
               if hasattr(action_dist, 'masked_log_prob') else \
               -(action_dist.log_prob(ac_flat) * mask_flat).sum() / mask_flat.sum()
        return loss

    def get_actions(self, img, obs, zero_std=True):
        policy_in = self._shared_forward(img, obs)
        return self._policy.get_actions(policy_in, zero_std=zero_std)
    
    def _shared_forward(self, imgs, obs):
        shared_in = torch.cat((self.embed(imgs), obs), dim=1) if self._use_obs \
                    else self.embed(imgs)
        return self._shared_mlp(shared_in)
    
    def embed(self, imgs):
        def embed_helper(net, im):
            if len(im.shape) == 5:
                B, T, C, H, W = im.shape
                embeds = net(im.reshape((B * T, C, H, W)))
                embeds = embeds.reshape((B, T * net.embed_dim))
                return embeds
            return net(im)
        
        if self._share_cam_features:
            embeds = [embed_helper(self.visual_features, imgs[f'cam{i}']) \
                      for i in range(self._n_cams)]
        else:
            embeds = [embed_helper(net, imgs[f'cam{i}']) \
                      for i, net in enumerate(self.visual_features)]
        return torch.cat(embeds, dim=1)

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
