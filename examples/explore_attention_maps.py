import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from data4robotics import load_vit


class TransformerAttentionWrapper(torch.nn.Module):
    def __init__(self, transformer, layer_num=11):
        super().__init__()
        self.model = transformer
        self.layer_num = layer_num

    def _encode_input(self, x):
        B = x.shape[0]
        x = self.model.patch_embed(x)
        x = x + self.model.pos_embed[:, 1:, :]
        cls_token = self.model.cls_token + self.model.pos_embed[:, :1, :]
        x = torch.cat((cls_token.expand(B, -1, -1), x), dim=1)
        return x

    def forward_attention(self, x, layer):
        x = self._encode_input(x)
        for i, block in enumerate(self.model.blocks):
            if i == layer:
                x = block.norm1(x)
                B, N, C = x.shape
                qkv = block.attn.qkv(x).reshape(B, N, 3, block.attn.num_heads, C // block.attn.num_heads).permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)
                attn = (q @ k.transpose(-2, -1)) * block.attn.scale
                return attn.softmax(dim=-1)
            x = block(x)
        return x


class Agent:
    def __init__(self, model, transform):
        self.model = model
        self.transform = transform
        self.attention_wrapper = TransformerAttentionWrapper(self.model)

    def preprocess_images(self, imgs, img_size=256):
        resized_imgs = [cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA) for img in imgs]
        resized_imgs = np.array(resized_imgs, dtype=np.uint8)
        img_tensor = torch.from_numpy(resized_imgs).float().permute((0, 3, 1, 2)) / 255
        return self.transform(img_tensor)[None].cuda()

    def get_attention(self, img, layer=11):
        obs = self.preprocess_images([img])[0]
        return self.attention_wrapper.forward_attention(obs, layer)


def generate_heatmap(image, attn, head=1, patch_size=16, img_size=224):
    attn_map = attn[0, head, 0, 1:].reshape(1, 1, img_size // patch_size, img_size // patch_size)
    resized_attn_map = F.interpolate(attn_map, scale_factor=patch_size, mode='bilinear').cpu().detach().numpy().squeeze()
    image = cv2.resize(image[:, :, ::-1], (img_size, img_size))
    cmap = plt.get_cmap('jet')
    heatmap = cmap(resized_attn_map / resized_attn_map.max())[:, :, :3] * 255
    heatmap_image = np.clip(0.8 * image + 0.2 * heatmap, 0, 255).astype(int)
    return heatmap_image[:, :, ::-1]


def main(vit_model_name, img_path):
    vit_transform, vit_model = load_vit(vit_model_name)
    vit_model.eval()

    agent = Agent(vit_model, vit_transform)
    image = cv2.imread(img_path)

    attn = agent.get_attention(image)
    heatmaps = [generate_heatmap(image, attn, head) for head in range(12)]

    fig, axes = plt.subplots(3, 4, figsize=(12, 8))
    for i in range(3):
        for j in range(4):
            idx = i * 4 + j
            axes[i, j].imshow(heatmaps[idx])
            axes[i, j].set_title(f"Head {idx}")
            axes[i, j].axis('off')

    plt.tight_layout()
    plt.suptitle("Test Heatmap")
    plt.savefig(f"./examples/heatmap_grid_test_{vit_model_name}.png")
    print(f"Heatmaps saved for {vit_model_name}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize transformer attention heatmaps.")
    parser.add_argument(
        "--vit_model_name",
        type=str,
        default="VC1_hrp",
        help="Name of the ViT model to load (default: VC1_hrp)"
    )
    parser.add_argument(
        "--img_path",
        type=str,
        default="./examples/test_img.png",
        help="Path to the image to visualize (default: ./examples/test_img.png)"
    )
    
    args = parser.parse_args()
    main(args.vit_model_name, args.img_path)
