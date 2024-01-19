# Copyright (c) Sudeep Dasari, 2023

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
from r3m import load_r3m
from torchvision import models
from torch.nn.modules.linear import Identity
from data4robotics.models.base import BaseModel


def _make_norm(norm_cfg):
    if norm_cfg['name'] == 'batch_norm':
        return nn.BatchNorm2d
    if norm_cfg['name'] == 'group_norm':
        num_groups = norm_cfg['num_groups']
        return lambda num_channels: nn.GroupNorm(num_groups, num_channels)
    raise NotImplementedError(f"Missing norm layer: {norm_cfg['name']}")


def _construct_resnet(size, norm, weights=None):
    if size == 18:
        w = models.ResNet18_Weights
        m = models.resnet18(norm_layer=norm)
    elif size == 34:
        w = models.ResNet34_Weights
        m = models.resnet34(norm_layer=norm)
    elif size == 50:
        w = models.ResNet50_Weights
        m = models.resnet50(norm_layer=norm)
    else:
        raise NotImplementedError(f'Missing size: {size}')
    
    if weights is not None:
        w = w.verify(weights).get_state_dict(progress=True)
        old_keys = list(w.keys())
        if norm is not nn.BatchNorm2d:
            w = {k:v for k, v in w.items() if 'running_mean' not in k \
                                           and 'running_var' not in k}
        m.load_state_dict(w)
    return m


class ResNet(BaseModel):
    def __init__(self, size, norm_cfg, weights=None, restore_path=''):
        norm_layer = _make_norm(norm_cfg)
        model = _construct_resnet(size, norm_layer, weights)
        model.fc = Identity()
        super().__init__(model, restore_path)
        self._size = size

    def forward(self, x):
        return self._model(x)

    @property
    def embed_dim(self):
        return {18: 512, 34: 512, 50: 2048}[self._size]


class R3M(ResNet):
    def __init__(self, size):
        nn.Module.__init__(self)
        self._model = load_r3m(f'resnet{size}').module.convnet.cpu()
        self._size = size
