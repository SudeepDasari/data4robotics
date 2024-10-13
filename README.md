# The Ingredients for Robotic Diffusion Transformers
[DiT-Policy](https://dit-policy.github.io)

This repository offers an implementation of our improved Diffusion Transformer Policy, which achieved State-of-the-Art manipulation results on bi-manual ALOHA robots and single-arm DROID Franka robots. This repo also allows easy use of our advanced pre-trained representations from [prior]([data4robotics.github.i](https://data4robotics.github.io)) [work](https://hrp-robot.github.io). We've succesfully deployed policies from this code on Franka robots (w/ [DROID](https://github.com/droid-dataset/droid/tree/main) and [MaNiMo](https://github.com/AGI-Labs/manimo)), [ALOHA](https://tonyzhaozh.github.io/aloha/) robots, and on [LEAP hands](https://www.leaphand.com). Check out our [eval scripts](eval_scripts/README.md) for more information. These policies can also be tested in simulation (see [Sim README](https://github.com/SudeepDasari/data4robotics/tree/dit_release/data4robotics/sim)).

If you find the this codebase or the diffusion transformer useful at all, please cite:
```
@inproceedings{dasari2024ditpi,
    title={The Ingredients for Robotic Diffusion Transformers},
    author = {Mohan Kumar Srirama and Sudeep Dasari and Shikhar Bahl and Abhinav Gupta},
    booktitle = {arXiv e-prints},
    year={2024},
}
```

And if you use the representations, please cite:
```
@inproceedings{dasari2023datasets,
      title={An Unbiased Look at Datasets for Visuo-Motor Pre-Training},
      author={Dasari, Sudeep and Srirama, Mohan Kumar and Jain, Unnat and Gupta, Abhinav},
      booktitle={Conference on Robot Learning},
      year={2023},
      organization={PMLR}
}

@inproceedings{kumar2024hrp,
    title={HRP: Human Affordances for Robotic Pre-Training},
    author = {Mohan Kumar Srirama and Sudeep Dasari and Shikhar Bahl and Abhinav Gupta},
    booktitle = {Proceedings of Robotics: Science and Systems},
    address  = {Delft, Netherlands},
    year = {2024},
}
```

## Installation
Our repository is easy to install using miniconda or anaconda:

```
conda env create -f env.yml
conda activate data4robotics
pip install git+https://github.com/AGI-Labs/robobuf.git
pip install git+https://github.com/facebookresearch/r3m.git
pip install -e ./
pre-commit install  # required for pushing back to the source git
```

## Training BC Policies
First, you're going to need to convert your training trajectories into our [robobuf](https://github.com/AGI-Labs/robobuf/tree/main) format (pseudo-code below). Check out some example ALOHA and DROID conversion code [here](https://github.com/AGI-Labs/r2d2_to_robobuf).

```
def _resize_and_encode(rgb_img, size=(256,256)):
    bgr_image = cv2.resize(bgr_image, size, interpolation=cv2.INTER_AREA)
    _, encoded = cv2.imencode(".jpg", bgr_image)
    return encoded

def convert_trajectories(input_trajs, out_path):
    out_buffer = []
    for traj in tqdm(input_trajs):
        out_traj = []
        for in_obs, in_ac, in_reward in enumerate(data):
            out_obs = dict(state=np.array(in_obs['state']).astype(np.float32),
                           enc_cam_0=_resize_and_encode(in_obs['image']))
            out_action = np.array(in_ac).astype(np.float32)
            out_reward = float(in_reward)
            out_traj.append((out_obs, out_action, out_reward))
        out_buffer.append(out_traj)

    with open(os.path.join(out_path, 'buf.pkl'), 'wb') as f:
        pkl.dump(out_trajs, f)
```

Once the conversion is complete, you can run the one of the example commands below:
```
# OUR MODEL

# Diffusion Transformer (DiT head) w/ ResNet tokenizer
python finetune.py exp_name=test agent=diffusion task=end_effector_r6 agent/features=resnet_gn agent.features.restore_path=/pat/to/resnet18/IN_1M_resnet18.pth  trainer=bc_cos_sched ac_chunk=100

# SOME EXAMPLE BASELINES

# Gaussian Mixture Model bc-policy with SOUP representations
python finetune.py exp_name=test agent.features.restore_path=/path/to/SOUP_1M_DH.pth buffer_path=/data/path/buffer.pkl

# Diffusion Policy (U-Net head) w/ HRP representations
python finetune.py exp_name=test agent=diffusion_unet task=end_effector_r6 agent/features=vit_base agent.features.restore_path=/path/to/IN_hrp.pth buffer_path=/data/path/buffer.pkl trainer=bc_cos_sched ac_chunk=16
```
This will result in a policy checkpoint saved in the `bc_finetune/<exp_name>` folder.

## Using Pre-Trained Features
You can easily download our pre-trained represenations using the provided script: `./download_features.sh`. You may also download the features individually on our [release website](https://www.cs.cmu.edu/~data4robotics/release/).

The features are very modular, and easy to use in your own code-base! Please refer to the [example code](https://github.com/SudeepDasari/data4robotics/blob/main/pretrained_networks_example.py) if you're interested.

## Policy Deployment (Sim and Real)

Detailed instructions and eval scripts for real world deployment are provided [here](https://github.com/SudeepDasari/data4robotics/tree/dit_release/eval_scripts). Similarly, you can reproduce our sim results, using the command/code provided [here](https://github.com/SudeepDasari/data4robotics/tree/dit_release/data4robotics/sim).
