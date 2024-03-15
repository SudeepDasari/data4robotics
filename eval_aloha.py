import argparse
from pathlib import Path
import time
import cv2
import hydra
import numpy as np
import torch
import yaml
import os
import json
from collections import deque


# aloha imports

import sys
sys.path.append("/home/huzheyuan/Desktop/language-dagger/src")
sys.path.append("/home/huzheyuan/Desktop/language-dagger/src/aloha_pro/aloha_scripts/")
from aloha_pro.aloha_scripts.real_env import make_real_env
from aloha_pro.aloha_scripts.robot_utils import move_grippers
from aloha_pro.aloha_scripts.constants import DT, PUPPET_GRIPPER_JOINT_OPEN


PRED_HORIZON = 48
EXP_WEIGHT = 0
PERIOD = 1.0 / 30  # stop the policy from running faster than 30Hz
GAMMA = 0.85


class Policy:
    def __init__(self, agent_path, model_name, temp_ensemble):
        with open(Path(agent_path, "agent_config.yaml"), "r") as f:
            config_yaml = f.read()
            agent_config = yaml.safe_load(config_yaml)
        with open(Path(agent_path, "obs_config.yaml"), "r") as f:
            config_yaml = f.read()
            obs_config = yaml.safe_load(config_yaml)
        with open(Path(agent_path, "ac_norm.json"), "r") as f:
            ac_norm_dict = json.load(f)
            loc, scale = ac_norm_dict['loc'], ac_norm_dict['scale']
            self.loc = np.array(loc).astype(np.float32)
            self.scale  = np.array(scale).astype(np.float32)

        agent = hydra.utils.instantiate(agent_config)
        save_dict = torch.load(Path(agent_path, model_name), map_location="cpu")
        agent.load_state_dict(save_dict['model'])
        self.agent = agent.eval().cuda()

        self.transform = hydra.utils.instantiate(obs_config["transform"])
        self.img_keys = obs_config['imgs']

        print(f"loaded agent from {agent_path}, at step: {save_dict['global_step']}")
        self.act_history = deque(maxlen=PRED_HORIZON)
        self.last_ac = None
        self.temp_ensemble = temp_ensemble
        self._last_time = None

    def _proc_images(self, img_dict, size=(256,256)):
        torch_imgs = dict()
        for i, k in enumerate(self.img_keys):
            bgr_img = img_dict[k][:,:,:3]
            bgr_img = cv2.resize(bgr_img, size, interpolation=cv2.INTER_AREA)
            rgb_img = bgr_img[:,:,::-1].copy()
            rgb_img = torch.from_numpy(rgb_img).float().permute((2, 0, 1)) / 255
            torch_imgs[f'cam{i}'] = self.transform(rgb_img)[None].cuda()
        return torch_imgs
    
    def _proc_state(self, qpos):
        return torch.from_numpy(qpos)[None].cuda()

    def _infer_policy(self, obs):
        img = self._proc_images(obs['images'])
        state = self._proc_state(obs['qpos'])

        with torch.no_grad():
            ac = self.agent.get_actions(img, state)
            ac = ac[0].cpu().numpy().astype(np.float32)[:PRED_HORIZON]
        
        # make sure the model predicted enough steps
        assert len(ac) >= PRED_HORIZON, "model did not return enough predictions!"
        return ac

    def _forward_ensemble(self, obs):
        ac = self._infer_policy(obs)
        self.act_history.append(ac)

        # handle temporal blending
        num_actions = len(self.act_history)
        curr_act_preds = np.stack(
                [
                    pred_actions[i]
                    for (i, pred_actions) in zip(
                        range(num_actions - 1, -1, -1), self.act_history
                    )
                ]
            )

        # more recent predictions get exponentially *less* weight than older predictions
        weights = np.exp(-EXP_WEIGHT * np.arange(num_actions))
        weights = weights / weights.sum()
        
        # return the weighted average across all predictions for this timestep
        return np.sum(weights[:, None] * curr_act_preds, axis=0)

    def _forward_chunked(self, obs):
        if not len(self.act_history):
            acs = self._infer_policy(obs)
            for ac in acs:
                self.act_history.append(ac)
        
        raw_ac =self.act_history.popleft()
        last_ac = self.last_ac if self.last_ac is not None \
                  else raw_ac
        self.last_ac = raw_ac

        # pop the oldest action
        return GAMMA * raw_ac + (1 - GAMMA) * last_ac

    def forward(self, obs):
        ac = self._forward_ensemble(obs) if self.temp_ensemble \
             else self._forward_chunked(obs)
        
        # denormalize the actions
        ac = ac * self.scale + self.loc
        
        # check effective HZ
        if self._last_time is not None:
            delta = time.time() - self._last_time
            if delta < PERIOD:
                time.sleep(PERIOD - delta)
            print('Effective HZ:', 1.0 / (time.time() - self._last_time))
        self._last_time = time.time()
        return ac


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--T", default=400)
    parser.add_argument("--temp_ensemble", default=False, action='store_true')
    parser.add_argument("--num_rollouts", default=50)
    args = parser.parse_args()

    agent_path = os.path.expanduser(os.path.dirname(args.checkpoint))
    model_name = args.checkpoint.split('/')[-1]
    policy = Policy(agent_path, model_name, args.temp_ensemble)

    env = make_real_env(init_node=True)
    obs = env.reset()
    
    for _ in range(args.T):
        ac = policy.forward(obs.observation)
        obs = env.step(ac)


if __name__ == "__main__":
    main()

