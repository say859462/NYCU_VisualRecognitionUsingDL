import argparse
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms
from tqdm import tqdm

from model import ImageClassificationModel
from train import generate_cross_attention_bbox_local_view


def _fmt_pred(name, probs, pred_idx):
    return f"{name}:{pred_idx} ({probs[pred_idx].item():.2f})"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config.json')
    parser.add_argument('--val_dir', type=str, default='./Dataset/data/val')
    parser.add_argument('--num_samples_per_class', type=int, default=3)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--save_dir', type=str,
                        default='./Plot/Attention_Outputs/current')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    with open(args.config, 'r') as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    model_path = args.model_path if args.model_path is not None else config['best_model_path']

    model = ImageClassificationModel(
        num_classes=config['num_classes'],
        pretrained=False,
        num_subcenters=config.get('num_subcenters', 3),
        embed_dim=config.get('embed_dim', 256)
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    preprocess_geo = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(448)
    ])
    preprocess_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    for class_id in tqdm(range(config['num_classes']), desc="Classes Processed"):
        class_dir = os.path.join(args.val_dir, str(class_id))
        if not os.path.exists(class_dir):
            continue

        class_save_dir = os.path.join(args.save_dir, str(class_id))
        os.makedirs(class_save_dir, exist_ok=True)

        all_images = [
            os.path.join(class_dir, f)
            for f in os.listdir(class_dir)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ]
        if not all_images:
            continue

        sampled_image_paths = random.sample(
            all_images, min(args.num_samples_per_class, len(all_images))
        )

        for img_path in sampled_image_paths:
            raw_img = Image.open(img_path).convert('RGB')
            cropped_img = preprocess_geo(raw_img)
            input_tensor = preprocess_tensor(
                cropped_img).unsqueeze(0).to(device)
            rgb_img = np.float32(cropped_img) / 255.0

            with torch.no_grad():
                local_tensor = generate_cross_attention_bbox_local_view(
                    model=model,
                    images=input_tensor,
                    threshold_ratio=config.get('local_crop_threshold', 0.45),
                    padding_ratio=config.get('local_crop_padding_ratio', 0.18),
                    min_crop_ratio=config.get('local_min_crop_ratio', 0.30),
                    max_crop_ratio=config.get('local_max_crop_ratio', 0.85),
                    fallback_crop_ratio=config.get(
                        'local_fallback_crop_ratio', 0.50),
                )

                outputs = model.forward_full_local(input_tensor, local_tensor)
                full_logits = outputs['full_logits']
                local_logits = outputs['local1_logits']
                fused_logits = outputs['fused_logits']

                full_probs = torch.softmax(full_logits, dim=1)[0]
                local_probs = torch.softmax(local_logits, dim=1)[0]
                fused_probs = torch.softmax(fused_logits, dim=1)[0]

                full_pred = full_probs.argmax().item()
                local_pred = local_probs.argmax().item()
                fused_pred = fused_probs.argmax().item()
                fused_score = fused_probs.max().item()

                attn_map = model.get_cross_attention_map(
                    input_tensor)   # [1,1,7,7]
                upsampled_attn = F.interpolate(
                    attn_map,
                    size=(448, 448),
                    mode='bilinear',
                    align_corners=False
                )

            heatmap = upsampled_attn.squeeze().cpu().numpy()
            heatmap = (heatmap - heatmap.min()) / \
                (heatmap.max() - heatmap.min() + 1e-8)
            vis = show_cam_on_image(rgb_img, heatmap, use_rgb=True)

            local_img = local_tensor[0].detach().cpu()
            local_img = local_img * \
                torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            local_img = local_img + \
                torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            local_img = torch.clamp(
                local_img, 0.0, 1.0).permute(1, 2, 0).numpy()

            fig, axes = plt.subplots(1, 3, figsize=(14, 5))
            title_color = 'green' if str(
                fused_pred) == str(class_id) else 'red'
            fig.suptitle(
                (
                    f"True: {class_id} | Pred(Fused): {fused_pred} "
                    f"(Conf: {fused_score * 100:.1f}%)\n"
                    f"{_fmt_pred('F', full_probs, full_pred)} | "
                    f"{_fmt_pred('L', local_probs, local_pred)} | "
                    f"{_fmt_pred('Fu', fused_probs, fused_pred)}"
                ),
                color=title_color,
                fontweight='bold'
            )

            axes[0].imshow(cropped_img)
            axes[0].axis('off')
            axes[0].set_title("Original")

            axes[1].imshow(vis)
            axes[1].axis('off')
            axes[1].set_title("CLS Cross-Attention (7x7)")

            axes[2].imshow(local_img)
            axes[2].axis('off')
            axes[2].set_title("Local Crop")

            save_name = f"attn_{os.path.splitext(os.path.basename(img_path))[0]}.png"
            plt.tight_layout()
            plt.savefig(os.path.join(class_save_dir, save_name),
                        bbox_inches='tight')
            plt.close(fig)


if __name__ == '__main__':
    main()
