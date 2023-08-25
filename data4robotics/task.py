# Copyright (c) Sudeep Dasari, 2023

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch, wandb, json, cv2, imageio, os
import numpy as np
from torch.utils.data import DataLoader, IterableDataset
from data4robotics.replay_buffer import IterableWrapper
_TEST_WORKERS = 4


def _build_data_loader(buffer, batch_size, num_workers, is_train=False):
    if is_train and not isinstance(buffer, IterableDataset):
        buffer = IterableWrapper(buffer)
    
    return DataLoader(buffer, batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=not isinstance(buffer, IterableDataset),
                              pin_memory=True,
                              persistent_workers=True,
                              drop_last=True,
                              worker_init_fn= lambda _: np.random.seed())


class DefaultTask:
    def __init__(self, train_buffer, test_buffer, n_cams, obs_dim, ac_dim,
                 batch_size, num_workers):
        self.n_cams, self.obs_dim, self.ac_dim = n_cams, obs_dim, ac_dim
        self.train_loader = _build_data_loader(train_buffer, batch_size, num_workers, 
                                               is_train=True)
        if test_buffer is not None:
            test_workers = min(num_workers, _TEST_WORKERS)
            self.test_loader = _build_data_loader(test_buffer, batch_size, test_workers)
    
    def eval(self, trainer, global_step):
        losses = []
        for batch in self.test_loader:
            with torch.no_grad():
                loss = trainer.training_step(batch, global_step)
                losses.append(loss.item())
        
        mean_val_loss = np.mean(losses)
        print(f'Step: {global_step}\tVal Loss: {mean_val_loss:.4f}')
        if wandb.run is not None:
            wandb.log({'eval/task_loss': mean_val_loss}, step=global_step)
