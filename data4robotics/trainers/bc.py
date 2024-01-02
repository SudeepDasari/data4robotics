# Copyright (c) Sudeep Dasari, 2023

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
from torch import nn
from data4robotics.trainers.base import BaseTrainer


class BehaviorCloning(BaseTrainer):
    def training_step(self, batch, global_step):
        (imgs, obs), actions, _ = batch
        imgs, obs, actions = [ar.to(self.device_id) for ar in \
                                                  (imgs, obs, actions)]

        ac_flat = actions.reshape((actions.shape[0], -1))
        loss = self.model(imgs, obs, ac_flat)
        self.log("bc_loss", global_step, loss.item())
        if self.is_train:
            self.log("lr", global_step, self.lr)
        return loss
