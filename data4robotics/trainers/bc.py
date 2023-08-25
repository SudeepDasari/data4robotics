# Copyright (c) Sudeep Dasari, 2023

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
from torch import nn
from data4robotics.trainers.base import BaseTrainer


class BehaviorCloning(BaseTrainer):
    def __init__(self, agent, device_id, lr=1e-4, weight_decay=1e-4):
        self._lr, self._weight_decay = lr, weight_decay
        super().__init__(agent, device_id)

    def training_step(self, batch, global_step):
        (imgs, obs), actions, _ = batch
        imgs, obs, actions = [ar.to(self.device_id) for ar in \
                                                  (imgs, obs, actions)]

        action_dist = self.model(imgs, obs)
        ac_flat = actions.reshape((actions.shape[0], -1))
        loss = -torch.mean(action_dist.log_prob(ac_flat)) 

        self.log("bc_loss", global_step, loss.item())
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self._lr, 
                                     weight_decay=self._weight_decay)
        return optimizer

    def _save_callback(self, save_path, _):
        # pickle and save the agent also
        torch.save('agent.pkl', agent)
