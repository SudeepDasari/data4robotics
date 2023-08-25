# Copyright (c) Sudeep Dasari, 2023

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from torchvision import transforms


def image_norm(size=224):
    return transforms.Compose([transforms.Resize((size, size)),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def get_transform_by_name(name, size=224):
    if name == 'preproc':
        return transforms.Compose([transforms.Resize((size, size), antialias=False),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    if name == 'basic':
        return transforms.Compose([transforms.RandomResizedCrop(size=size, scale=(0.2, 1.0), antialias=False),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    if name == 'medium':
        kernel_size = int(0.05 * size); kernel_size = kernel_size + (1 - kernel_size % 2)
        return transforms.Compose([transforms.RandomResizedCrop(size=size, scale=(0.2, 1.0), antialias=False),
                                   transforms.GaussianBlur(kernel_size=kernel_size),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    if name == 'advanced':
        kernel_size = int(0.05 * size); kernel_size = kernel_size + (1 - kernel_size % 2)
        color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        return transforms.Compose([transforms.RandomResizedCrop(size=size, scale=(0.2, 1.0), antialias=False),
                                   transforms.RandomApply([color_jitter], p=0.8),
                                   transforms.RandomGrayscale(p=0.2),
                                   transforms.GaussianBlur(kernel_size=kernel_size),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    raise NotImplementedError(f'{name} not found!')
