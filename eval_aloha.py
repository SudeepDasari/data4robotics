import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import hydra
import numpy as np
import torch
import yaml

# aloha imports

sys.path.append("/home/huzheyuan/Desktop/language-dagger/src")
sys.path.append("/home/huzheyuan/Desktop/language-dagger/src/aloha_pro/aloha_scripts/")
from aloha_pro.aloha_scripts.constants import DT, PUPPET_GRIPPER_JOINT_OPEN
from aloha_pro.aloha_scripts.real_env import make_real_env
from aloha_pro.aloha_scripts.robot_utils import move_grippers


class Policy:
    def __init__(self, agent_path, model_name, args):
        self.args = args

        with open(Path(agent_path, "agent_config.yaml"), "r") as f:
            config_yaml = f.read()
            agent_config = yaml.safe_load(config_yaml)
        with open(Path(agent_path, "obs_config.yaml"), "r") as f:
            config_yaml = f.read()
            obs_config = yaml.safe_load(config_yaml)
        with open(Path(agent_path, "ac_norm.json"), "r") as f:
            ac_norm_dict = json.load(f)
            loc, scale = ac_norm_dict["loc"], ac_norm_dict["scale"]
            self.loc = np.array(loc).astype(np.float32)
            self.scale = np.array(scale).astype(np.float32)

        agent = hydra.utils.instantiate(agent_config)

        save_dict = torch.load(Path(agent_path, model_name), map_location="cpu")
        agent.load_state_dict(save_dict["model"])

        agent = torch.compile(agent)

        self.agent = agent.eval().cuda()

        self.transform = hydra.utils.instantiate(obs_config["transform"])
        self.img_keys = obs_config["imgs"]

        print(f"loaded agent from {agent_path}, at step: {save_dict['global_step']}")
        self.temp_ensemble = args.temp_ensemble
        self.pred_horizon = args.pred_horizon
        self.reset()

    def reset(self):
        self.act_history = deque(maxlen=self.pred_horizon)
        self.last_ac = None
        self._last_time = None

    def _proc_images(self, img_dict, size=(256, 256)):
        torch_imgs = dict()
        for i, k in enumerate(self.img_keys):
            bgr_img = img_dict[k][:, :, :3]
            bgr_img = cv2.resize(bgr_img, size, interpolation=cv2.INTER_AREA)
            rgb_img = bgr_img[:, :, ::-1].copy()
            rgb_img = torch.from_numpy(rgb_img).float().permute((2, 0, 1)) / 255
            torch_imgs[f"cam{i}"] = self.transform(rgb_img)[None].cuda()
        return torch_imgs

    def _proc_state(self, qpos):
        return torch.from_numpy(qpos).float()[None].cuda()

    def _infer_policy(self, obs):
        import time

        start = time.time()
        img = self._proc_images(obs["images"])
        print("Image processing time:", time.time() - start)
        start = time.time()
        state = self._proc_state(obs["qpos"])
        print("State processing time:", time.time() - start)

        start = time.time()
        with torch.no_grad():
            ac = self.agent.get_actions(img, state)
            ac = ac[0].cpu().numpy().astype(np.float32)[: self.args.pred_horizon]
        print("Inference time:", time.time() - start)

        # make sure the model predicted enough steps
        assert len(ac) >= self.args.pred_horizon, "model did not return enough predictions!"
        return ac

    def _forward_ensemble(self, obs):
        ac = self._infer_policy(obs)
        self.act_history.append(ac)

        # potentially consider not ensembling every timestep.

        # handle temporal blending
        num_actions = len(self.act_history)
        print("Num actions:", num_actions)
        curr_act_preds = np.stack(
            [pred_actions[i] for (i, pred_actions) in zip(range(num_actions - 1, -1, -1), self.act_history)]
        )

        # more recent predictions get exponentially *less* weight than older predictions
        weights = np.exp(-self.args.exp_weight * np.arange(num_actions))
        weights = weights / weights.sum()

        # return the weighted average across all predictions for this timestep
        return np.sum(weights[:, None] * curr_act_preds, axis=0)

    def _forward_chunked(self, obs):
        if not len(self.act_history):
            acs = self._infer_policy(obs)
            for ac in acs:
                self.act_history.append(ac)

        raw_ac = self.act_history.popleft()
        last_ac = self.last_ac if self.last_ac is not None else raw_ac
        self.last_ac = self.args.gamma * raw_ac + (1 - self.args.gamma) * last_ac
        return self.last_ac.copy()

    def forward(self, obs):
        ac = self._forward_ensemble(obs) if self.temp_ensemble else self._forward_chunked(obs)

        # denormalize the actions
        ac = ac * self.scale + self.loc

        # check effective HZ
        if self._last_time is not None:
            delta = time.time() - self._last_time
            if delta < self.args.period:
                time.sleep(self.args.period - delta)
            print("Effective HZ:", 1.0 / (time.time() - self._last_time))
        self._last_time = time.time()
        return ac


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--T", default=400, type=int)
    parser.add_argument("--temp_ensemble", default=False, action="store_true")
    parser.add_argument("--num_rollouts", default=1, type=int)
    parser.add_argument("--pred_horizon", default=48, type=int)
    parser.add_argument("--exp_weight", default=0, type=float)
    parser.add_argument("--hz", default=48, type=float)
    parser.add_argument("--gamma", default=0.85, type=float)

    args = parser.parse_args()
    args.period = 1.0 / args.hz

    agent_path = os.path.expanduser(os.path.dirname(args.checkpoint))
    model_name = args.checkpoint.split("/")[-1]
    policy = Policy(agent_path, model_name, args)

    env = make_real_env(init_node=True)

    # Roll out the policy num_rollout times
    for _ in range(args.num_rollouts):

        last_input = None
        while last_input != 'y':
            if last_input == 'r':
                obs = env.reset()
            last_input = input("Continue with rollout (y; r to reset now)?")
        
        policy.reset()
        obs = env.reset()

        for _ in range(args.T):
            ac = policy.forward(obs.observation)
            obs = env.step(ac)

        # Reset gripper to let go of stuff
        env._reset_gripper()

if __name__ == "__main__":
    main()
