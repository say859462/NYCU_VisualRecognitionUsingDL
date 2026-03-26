import argparse
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from model import ImageClassificationModel


def _denorm(tensor):
    img = tensor.detach().cpu()
    img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    return torch.clamp(img, 0.0, 1.0).permute(1, 2, 0).numpy()


def _normalize_map(x: torch.Tensor) -> torch.Tensor:
    x = x - x.amin(dim=(-2, -1), keepdim=True)
    x = x / (x.amax(dim=(-2, -1), keepdim=True) + 1e-8)
    return x


def _smooth_heatmap(saliency: torch.Tensor, out_size=(448, 448)) -> torch.Tensor:
    """
    saliency: [B, H, W] or [B,1,H,W]
    return:   [B, H_out, W_out]
    """
    if saliency.dim() == 3:
        saliency = saliency.unsqueeze(1)  # [B,1,H,W]

    saliency = _normalize_map(saliency)

    # 先用 bicubic 放大，減少 blocky 感
    saliency = F.interpolate(
        saliency,
        size=out_size,
        mode="bicubic",
        align_corners=False
    )

    # 再做幾次平滑，讓 heatmap 更像連續熱區
    saliency = F.avg_pool2d(saliency, kernel_size=7, stride=1, padding=3)
    saliency = F.avg_pool2d(saliency, kernel_size=7, stride=1, padding=3)

    saliency = _normalize_map(saliency)
    return saliency.squeeze(1)


def _overlay_heatmap_on_image(rgb_img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45):
    """
    rgb_img: [H,W,3], range [0,1]
    heatmap: [H,W], range [0,1]
    """
    cmap = plt.get_cmap("jet")
    heatmap_color = cmap(heatmap)[..., :3]  # 去掉 alpha channel
    overlay = (1 - alpha) * rgb_img + alpha * heatmap_color
    overlay = np.clip(overlay, 0.0, 1.0)
    return overlay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config.json')
    parser.add_argument('--val_dir', type=str, default='./Dataset/data/val')
    parser.add_argument('--num_samples_per_class', type=int, default=3)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='./Plot/Attention_Outputs/70th')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

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

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
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
            all_images,
            min(args.num_samples_per_class, len(all_images))
        )

        for img_path in sampled_image_paths:
            raw_img = Image.open(img_path).convert('RGB')
            cropped_img = preprocess_geo(raw_img)
            input_tensor = preprocess_tensor(cropped_img).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model.forward_pmg(input_tensor)

                global_probs = torch.softmax(outputs['global_logits'], dim=1)[0]
                part2_probs = torch.softmax(outputs['part2_logits'], dim=1)[0]
                part4_probs = torch.softmax(outputs['part4_logits'], dim=1)[0]
                concat_probs = torch.softmax(outputs['concat_logits'], dim=1)[0]

                # 原始 saliency
                saliency = model.get_saliency(input_tensor)   # [B,H,W] or [B,1,H,W]

                # 平滑版 heatmap
                saliency_smooth = _smooth_heatmap(saliency, out_size=(448, 448))[0].cpu().numpy()

            rgb_img = np.float32(cropped_img) / 255.0
            overlay = _overlay_heatmap_on_image(rgb_img, saliency_smooth, alpha=0.45)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(
                f"True: {class_id} | "
                f"G:{global_probs.argmax().item()}({global_probs.max().item():.2f}) | "
                f"P2:{part2_probs.argmax().item()}({part2_probs.max().item():.2f}) | "
                f"P4:{part4_probs.argmax().item()}({part4_probs.max().item():.2f}) | "
                f"C:{concat_probs.argmax().item()}({concat_probs.max().item():.2f})",
                fontweight='bold'
            )

            axes[0].imshow(cropped_img)
            axes[0].axis('off')
            axes[0].set_title('Original')

            axes[1].imshow(saliency_smooth, cmap='jet')
            axes[1].axis('off')
            axes[1].set_title('Smoothed Heatmap')

            axes[2].imshow(overlay)
            axes[2].axis('off')
            axes[2].set_title('Overlay')

            save_name = f"pmg_heatmap_{os.path.splitext(os.path.basename(img_path))[0]}.png"
            plt.tight_layout()
            plt.savefig(
                os.path.join(class_save_dir, save_name),
                bbox_inches='tight',
                dpi=200
            )
            plt.close(fig)


if __name__ == '__main__':
    main()