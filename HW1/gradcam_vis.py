import torch
import os
import random
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from model import ImageClassificationModel
from pytorch_grad_cam.utils.image import show_cam_on_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config.json')
    parser.add_argument('--val_dir', type=str, default='./Dataset/data/val')
    parser.add_argument('--num_samples_per_class', type=int, default=3)
    parser.add_argument('--model_path', type=str,
                        default='./Model_Weight/best_model.pth')
    parser.add_argument('--save_dir', type=str,
                        default='./Plot/Attention_Outputs/65th')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    model = ImageClassificationModel(
        num_classes=config['num_classes'],
        pretrained=False,
        num_subcenters=config.get('num_subcenters', 3),
        embed_dim=config.get('embed_dim', 256)
    ).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    preprocess_geo = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(448)
    ])
    preprocess_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
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
                logits = model(input_tensor)
                probs = torch.softmax(logits, dim=1)[0]
                pred_class = probs.argmax().item()
                pred_score = probs.max().item()

                # 真正的 cross-attention map
                attn_map = model.get_cross_attention_map(
                    input_tensor)   # [1,1,4,4]
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

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            color = 'green' if str(pred_class) == str(class_id) else 'red'
            fig.suptitle(
                f"True: {class_id} | Pred: {pred_class} (Conf: {pred_score*100:.1f}%)",
                color=color,
                fontweight='bold'
            )

            axes[0].imshow(cropped_img)
            axes[0].axis('off')
            axes[0].set_title("Original")

            axes[1].imshow(vis)
            axes[1].axis('off')
            axes[1].set_title("CLS Cross-Attention")

            save_name = f"attn_{os.path.splitext(os.path.basename(img_path))[0]}.png"
            plt.savefig(os.path.join(class_save_dir, save_name),
                        bbox_inches='tight')
            plt.close(fig)


if __name__ == '__main__':
    main()
