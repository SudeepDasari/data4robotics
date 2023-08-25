# Copyright (c) Sudeep Dasari, 2023

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import random, torch, tqdm
import numpy as np
import pickle as pkl
from robobuf import ReplayBuffer as RB
from torch.utils.data import Dataset, IterableDataset


# helper functions
_img_to_tensor = lambda x: torch.from_numpy(x.copy()).permute((0, 3, 1, 2)).float() / 255
_to_tensor = lambda x: torch.from_numpy(x).float()


BUF_SHUFFLE_RNG = 3904767649
class ReplayBuffer(Dataset):
    def __init__(self, buffer_path, transform=None, n_train_demos=200, mode='train', ac_chunk=1):
        assert mode in ('train', 'test'), "Mode must be train/test"
        buffer_data = self._load_buffer(buffer_path)
        assert len(buffer_data) >= n_train_demos, "Not enough demos!"

        # shuffle the list with the fixed seed
        rng = random.Random(BUF_SHUFFLE_RNG)
        rng.shuffle(buffer_data)

        # split data according to mode
        buffer_data = buffer_data[:n_train_demos] if mode == 'train' \
                      else buffer_data[n_train_demos:]
        
        self.transform = transform
        self.s_a_sprime = []
        for traj in tqdm.tqdm(buffer_data):
            imgs, obs, acs = traj['images'], traj['observations'], traj['actions']
            assert len(obs) == len(acs) and len(acs) == len(imgs), "All time dimensions must match!"

            # pad camera dimension if needed
            if len(imgs.shape) == 4:
                imgs = imgs[:,None]

            for t in range(len(imgs) - ac_chunk):
                i_t, o_t = imgs[t], obs[t]
                i_t_prime, o_t_prime = imgs[t+ac_chunk], obs[t+ac_chunk]
                a_t = acs[t:t+ac_chunk]
                self.s_a_sprime.append(((i_t, o_t), a_t, (i_t_prime, o_t_prime)))
    
    def _load_buffer(self, buffer_path):
        print('loading', buffer_path)
        with open(buffer_path, 'rb') as f:
            buffer_data = pkl.load(f)
        return buffer_data

    def __len__(self):
        return len(self.s_a_sprime)
    
    def __getitem__(self, idx):
        (i_t, o_t), a_t, (i_t_prime, o_t_prime) = self.s_a_sprime[idx]

        i_t, i_t_prime = _img_to_tensor(i_t), _img_to_tensor(i_t_prime)
        o_t, a_t, o_t_prime = _to_tensor(o_t), _to_tensor(a_t), _to_tensor(o_t_prime)

        if self.transform is not None:
            N_CAM = i_t.shape[0]
            imgs = torch.cat((i_t, i_t_prime), dim=0)
            imgs = self.transform(imgs)
            i_t, i_t_prime = imgs[:N_CAM], imgs[N_CAM:]
        return (i_t, o_t), a_t, (i_t_prime, o_t_prime)


def _embed_img(features, img, device, transform):
    img = transform(_img_to_tensor(img))
    with torch.no_grad():
        feat = features(img.to(device))
    return feat.reshape(-1).cpu().numpy().astype(np.float32)


class IterableWrapper(IterableDataset):
    def __init__(self, wrapped_dataset, max_count=float('inf')):
        self.wrapped = wrapped_dataset
        self.ctr, self.max_count = 0, max_count
    
    def __iter__(self):
        self.ctr = 0
        return self
    
    def __next__(self):
        if self.ctr > self.max_count:
            raise StopIteration
        
        self.ctr += 1
        idx = int(np.random.choice(len(self.wrapped)))
        return self.wrapped[idx]


class RobobufReplayBuffer(ReplayBuffer):
    def __init__(self, buffer_path, transform=None, n_test_trans=500, mode='train', ac_chunk=1, cam_idx=0):
        assert mode in ('train', 'test'), "Mode must be train/test"
        with open(buffer_path, 'rb') as f:
            buf = RB.load_traj_list(pkl.load(f))
        assert len(buf) > n_test_trans, "Not enough transitions!"
        assert ac_chunk == 1, "Only supports ac_chunk of 1 for now!"

        # shuffle the list with the fixed seed
        rng = random.Random(BUF_SHUFFLE_RNG)

        # get and shuffle list of buf indices
        index_list = list(range(len(buf)))
        rng.shuffle(index_list)

        # split data according to mode
        index_list = index_list[n_test_trans:] if mode == 'train' \
                     else index_list[:n_test_trans]
        
        self.transform = transform
        self.s_a_sprime = []
        last = 0
        print(f'Building {mode} buffer with cam_idx={cam_idx}')
        for idx in tqdm.tqdm(index_list):
            t = buf[idx]
            if t.next is None:
                last += 1
                continue

            i_t, o_t = t.obs.image(cam_idx)[None], t.obs.state
            i_t_prime, o_t_prime = t.next.obs.image(cam_idx)[None], t.next.obs.state
            a_t = t.action
            self.s_a_sprime.append(((i_t, o_t), a_t, (i_t_prime, o_t_prime)))
