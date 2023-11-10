# Copyright (c) Sudeep Dasari, 2023

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os, torch, subprocess, data4robotics
from torchvision import transforms
from data4robotics.models import vit
from data4robotics.models import resnet


# feature install path
BASE_PATH = os.path.dirname(data4robotics.__file__) + "/../"
FEATURE_PATH = os.path.join(BASE_PATH, "visual_features")


def _check_and_download():
    download_script = os.path.join(BASE_PATH, "download_features.sh")
    subprocess.run([download_script, BASE_PATH], check=True)


def default_transform():
    return transforms.Compose([transforms.Resize((224, 224), antialias=False),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                    std=[0.229, 0.224, 0.225])])


def load_vit(model_name="SOUP_1M_DH", device=torch.device('cuda:0')):
    _check_and_download()
    model = vit.vit_base_patch16(img_size=224, use_cls=True, drop_path_rate=0.0)
    restore_path = os.path.join(FEATURE_PATH, f'vit_base/{model_name}.pth')
    model = vit.load_vit(model, restore_path)
    return default_transform(), model.to(device)


def load_resnet18(model_name="IN_1M_resnet18", device=torch.device('cuda:0')):
    _check_and_download()
    restore_path = os.path.join(FEATURE_PATH, f'resnet18/{model_name}.pth')
    model = resnet.ResNet(size=18, pretrained=None, restore_path=restore_path,
                          norm_cfg=dict(name='group_norm', num_groups=16))
    return default_transform(), model.to(device)
