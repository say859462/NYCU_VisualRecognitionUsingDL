import torch
import os
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F  # ⭐ 新增
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from model import ImageClassificationModel
from pytorch_grad_cam.utils.image import show_cam_on_image  # 僅保留疊圖工具


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_dir', type=str, default='./Dataset/data/val')
    parser.add_argument('--num_samples_per_class', type=int, default=3)
    parser.add_argument('--model_path', type=str,
                        default='./Model_Weight/best_model.pth')
    parser.add_argument('--save_dir', type=str,
                        default='./Plot/Attention_Outputs/53th')  # ⭐ 建議改名
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    model = ImageClassificationModel(
        num_classes=100, pretrained=False).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # ⭐ 刪除 cam = GradCAM(...) 的宣告，我們不需要它了

    preprocess_geo = transforms.Compose(
        [transforms.Resize(576), transforms.CenterCrop(512)])
    preprocess_tensor = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    for class_id in tqdm(range(100), desc="Classes Processed"):
        class_dir = os.path.join(args.val_dir, str(class_id))
        if not os.path.exists(class_dir):
            continue

        class_save_dir = os.path.join(args.save_dir, str(class_id))
        os.makedirs(class_save_dir, exist_ok=True)

        all_images = [os.path.join(class_dir, f) for f in os.listdir(
            class_dir) if f.lower().endswith(('.jpg', '.png'))]
        if not all_images:
            continue

        sampled_image_paths = random.sample(all_images, min(
            args.num_samples_per_class, len(all_images)))

        for img_path in sampled_image_paths:
            raw_img = Image.open(img_path).convert('RGB')
            cropped_img = preprocess_geo(raw_img)
            input_tensor = preprocess_tensor(
                cropped_img).unsqueeze(0).to(device)
            rgb_img = np.float32(cropped_img) / 255.0

            with torch.no_grad():
                logits = model(input_tensor)
                probs = torch.nn.functional.softmax(logits, dim=1)[0]
                pred_class, pred_score = probs.argmax().item(), probs.max().item()

                raw_saliency = model.get_saliency(input_tensor)

                raw_saliency = raw_saliency.unsqueeze(1)
                upsampled_saliency = F.interpolate(
                    raw_saliency, size=(512, 512), mode='bilinear', align_corners=False
                )

            # 轉換為 Numpy 並進行 Min-Max 正規化至 0~1
            saliency_map = upsampled_saliency.squeeze().cpu().numpy()
            saliency_map = (saliency_map - saliency_map.min()) / \
                (saliency_map.max() - saliency_map.min() + 1e-8)

            # 使用 GradCAM 的繪圖工具進行熱力圖疊加
            vis = show_cam_on_image(rgb_img, saliency_map, use_rgb=True)

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            color = 'green' if str(pred_class) == str(class_id) else 'red'
            fig.suptitle(
                f"True: {class_id} | Pred: {pred_class} (Conf: {pred_score*100:.1f}%)", color=color, fontweight='bold')

            axes[0].imshow(cropped_img)
            axes[0].axis('off')
            axes[0].set_title("Original")
            axes[1].imshow(vis)
            axes[1].axis('off')
            axes[1].set_title("Native Attention Pooling")

            plt.savefig(os.path.join(
                class_save_dir, f"attn_{os.path.splitext(os.path.basename(img_path))[0]}.png"))
            plt.close(fig)


if __name__ == '__main__':
    main()
