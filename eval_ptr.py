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

# r2d2 robot imports
from r2d2.user_interface.eval_gui import EvalGUI


class BaselinePolicy:
    def __init__(self, agent_path, model_name):
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
        self.img_key = obs_config['img']

        print(f"loaded agent from {agent_path}, at step: {save_dict['global_step']}")
        self._last_time = None
    
    def _proc_image(self, zed_img, size=(256,256)):
        bgr_img = zed_img[:,:,:3]
        bgr_img = cv2.resize(bgr_img, size, interpolation=cv2.INTER_AREA)
        rgb_img = bgr_img[:,:,::-1].copy()
        rgb_img = torch.from_numpy(rgb_img).float().permute((2, 0, 1)) / 255
        return {'cam0': self.transform(rgb_img)[None].cuda()}
    
    def _proc_state(self, cart_pos, grip_pos):
        state = np.concatenate((cart_pos, np.array([grip_pos]))).astype(np.float32)
        return torch.from_numpy(state)[None].cuda()

    def forward(self, obs):
        img = self._proc_image(obs['image'][self.img_key])
        state = self._proc_state(obs['robot_state']['cartesian_position'],
                                 obs['robot_state']['gripper_position'])
        with torch.no_grad():
            ac = self.agent.get_actions(img, state)
        ac = ac[0,0].cpu().numpy().astype(np.float32)
        ac = np.clip(ac * self.scale + self.loc, -1, 1)    # denormalize the actions

        cur_time = time.time()
        if self._last_time is not None:
            print('Effective HZ:', 1.0 / (cur_time - self._last_time))
        self._last_time = cur_time
        return ac
    
    def load_goal_imgs(self, goal_dict):
        pass

    def load_lang(self, text):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    args = parser.parse_args()

    agent_path = os.path.expanduser(os.path.dirname(args.checkpoint))
    model_name = args.checkpoint.split('/')[-1]
    policy = BaselinePolicy(agent_path, model_name)

    # test with a null observation if you desire
    # img = {'26638268_left':np.zeros((512, 512, 3))}
    # rbt_state = dict(cartesian_position=np.zeros((6,)), 
    #                  gripper_position=0)
    # null_obs = dict(image=img, robot_state=rbt_state)
    # policy.forward(null_obs); policy.forward(null_obs)
    # exit(0)

    # start up R2D2 eval gui
    EvalGUI(policy=policy)


if __name__ == "__main__":
    main()
