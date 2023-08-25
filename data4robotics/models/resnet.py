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
        return models.resnet18(weights=weights, norm_layer=norm)
    if size == 34:
        return models.resnet34(weights=weights, norm_layer=norm)
    if size == 50:
        return models.resnet34(weights=weights, norm_layer=norm)
    raise NotImplementedError(f'Missing size: {size}')


class ResNet(BaseModel):
    def __init__(self, size, norm_cfg, pretrained=None, restore_path=''):
        norm_layer = _make_norm(norm_cfg)
        model = _construct_resnet(size, norm_layer, pretrained)
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
