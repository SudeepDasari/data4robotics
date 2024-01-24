# Copyright (c) Sudeep Dasari, 2023

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import random, torch, tqdm, os, shutil
import numpy as np
import pickle as pkl
from robobuf import ReplayBuffer as RB
from torch.utils.data import Dataset, IterableDataset


# helper functions
_img_to_tensor = lambda x: torch.from_numpy(x.copy()).permute((0, 3, 1, 2)).float() / 255
_to_tensor = lambda x: torch.from_numpy(x).float()
def _get_imgs(t, cam_idx, past_frames):
    imgs = []
    while len(imgs) < past_frames + 1:
        imgs.append(t.obs.image(cam_idx)[None])

        if t.prev is not None:
            t = t.prev
    return np.concatenate(imgs, axis=0)


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
        self.s_a_mask = []
        for traj in tqdm.tqdm(buffer_data):
            imgs, obs, acs = traj['images'], traj['observations'], traj['actions']
            assert len(obs) == len(acs) and len(acs) == len(imgs), "All time dimensions must match!"

            # pad camera dimension if needed
            if len(imgs.shape) == 4:
                imgs = imgs[:,None]

            for t in range(len(imgs) - ac_chunk):
                i_t = {f'cam{c}': imgs[t,c] for c in range(imgs.shape[1])}
                loss_mask = np.ones((ac_chunk,), dtype=np.float32)
                o_t, a_t = obs[t], acs[t:t+ac_chunk]
                self.s_a_mask.append(((i_t, o_t), a_t, loss_mask))
    
    def _load_buffer(self, buffer_path):
        print('loading', buffer_path)
        with open(buffer_path, 'rb') as f:
            buffer_data = pkl.load(f)
        return buffer_data

    def __len__(self):
        return len(self.s_a_mask)
    
    def __getitem__(self, idx):
        (i_t, o_t), a_t, loss_mask = self.s_a_mask[idx]

        i_t = {k: _img_to_tensor(v) for k, v in i_t.items()}
        if self.transform is not None:
            i_t = {k: self.transform(v) for k, v in i_t.items()}

        o_t, a_t = _to_tensor(o_t), _to_tensor(a_t)
        loss_mask = _to_tensor(loss_mask)[:,None].repeat((1 ,a_t.shape[-1]))
        assert loss_mask.shape[0] == a_t.shape[0], "a_t and mask shape must be ac_chunk!"
        return (i_t, o_t), a_t, loss_mask


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
    def __init__(self, buffer_path, transform=None, n_test_trans=500, mode='train', 
                 ac_chunk=1, cam_indexes=[0], past_frames=0):
        assert mode in ('train', 'test'), "Mode must be train/test"
        with open(buffer_path, 'rb') as f:
            buf = RB.load_traj_list(pkl.load(f))
        assert len(buf) > n_test_trans, "Not enough transitions!"

        norm_file = os.path.join(os.path.dirname(buffer_path), 'ac_norm.json')
        if os.path.exists(norm_file):
            shutil.copyfile(norm_file, './ac_norm.json')

        # shuffle the list with the fixed seed
        rng = random.Random(BUF_SHUFFLE_RNG)

        # get and shuffle list of buf indices
        index_list = list(range(len(buf)))
        rng.shuffle(index_list)

        # split data according to mode
        index_list = index_list[n_test_trans:] if mode == 'train' \
                     else index_list[:n_test_trans]
        
        self.transform = transform
        self.s_a_mask = []
        print(f'Building {mode} buffer with cam_indexes={cam_indexes}')
        for idx in tqdm.tqdm(index_list):
            t = buf[idx]

            loop_t, chunked_actions, loss_mask = t, [], []
            for _ in range(ac_chunk):
                chunked_actions.append(loop_t.action[None])
                loss_mask.append(1.0)
                loop_t = loop_t.next

                if loop_t is None:
                    break

            if len(chunked_actions) < ac_chunk:
                for _ in range(ac_chunk - len(chunked_actions)):
                    chunked_actions.append(chunked_actions[-1])
                    loss_mask.append(0.0)

            i_t = {f'cam{i}': _get_imgs(t, cam_idx, past_frames) \
                    for i, cam_idx in enumerate(cam_indexes)}
            o_t = t.obs.state
            a_t = np.concatenate(chunked_actions, 0).astype(np.float32)
            loss_mask = np.array(loss_mask, dtype=np.float32)
            self.s_a_mask.append(((i_t, o_t), a_t, loss_mask))
