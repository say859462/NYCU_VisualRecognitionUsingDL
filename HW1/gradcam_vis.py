import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import random
import argparse
from torchvision import transforms
from tqdm import tqdm
import torch.nn.functional as F  # ⭐ 新增：用於特徵圖插值放大

# 載入你寫好的模型
from model import ImageClassificationModel

# 載入 Grad-CAM 套件
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def main():
    parser = argparse.ArgumentParser(
        description="Dual-layer Grad-CAM & RSA Visualization for All Classes")
    parser.add_argument('--val_dir', type=str, default='./Dataset/data/val',
                        help='Path to the validation dataset root directory')
    parser.add_argument('--num_samples_per_class', type=int, default=3,
                        help='Number of images to randomly sample per class')

    parser.add_argument('--model_path', type=str,
                        default='./Model_Weight/best_model.pth', help='Path to model weight')

    parser.add_argument('--num_classes', type=int,
                        default=100, help='Number of classes')

    parser.add_argument('--save_dir', type=str,
                        default='./Plot/GradCAM_Outputs/22th', help='Directory to save heatmaps')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 確保主儲存目錄存在
    os.makedirs(args.save_dir, exist_ok=True)

    # 2. 初始化模型並載入權重
    model = ImageClassificationModel(
        num_classes=args.num_classes, pretrained=False).to(device)
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Loaded model weight from: {args.model_path}")
    else:
        raise FileNotFoundError(f"Model weight not found at {args.model_path}")

    model.eval()

    # 3. 初始化 Layer 3 的 Grad-CAM (淺層看紋理，依然好用且不受 CBP 嚴重污染)
    cam_l3 = GradCAM(model=model, target_layers=[model.reduce3])

    # 影像前處理定義
    preprocess_geo = transforms.Compose([
        transforms.Resize(600),
        transforms.CenterCrop(512)
    ])
    preprocess_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

    print(f"\nProcessing all {args.num_classes} classes...")
    for class_id in tqdm(range(args.num_classes), desc="Classes Processed"):
        class_dir = os.path.join(args.val_dir, str(class_id))

        if not os.path.exists(class_dir):
            continue

        class_save_dir = os.path.join(args.save_dir, str(class_id))
        os.makedirs(class_save_dir, exist_ok=True)

        all_images = [
            os.path.join(class_dir, f) for f in os.listdir(class_dir)
            if f.lower().endswith(valid_extensions) and os.path.isfile(os.path.join(class_dir, f))
        ]

        if not all_images:
            continue

        actual_samples = min(args.num_samples_per_class, len(all_images))
        sampled_image_paths = random.sample(all_images, actual_samples)

        for img_path in sampled_image_paths:
            try:
                raw_img = Image.open(img_path).convert('RGB')
                cropped_img = preprocess_geo(raw_img)
                input_tensor = preprocess_tensor(
                    cropped_img).unsqueeze(0).to(device)
                rgb_img = np.float32(cropped_img) / 255.0

                # 取得預測結果與 ⭐內建 RSA 空間權重
                with torch.no_grad():
                    # 啟動 return_attn=True，拿回 spatial_attn (Shape: [1, 1, 14, 14])
                    outputs, spatial_attn = model(
                        input_tensor, return_attn=True)

                    T = 2.5
                    probabilities = torch.nn.functional.softmax(
                        outputs / T, dim=1)[0]

                    pred_class = probabilities.argmax().item()
                    pred_score = probabilities[pred_class].item()

                # 執行 Layer 3 Grad-CAM
                grayscale_cam_l3 = cam_l3(
                    input_tensor=input_tensor, targets=None)[0, :]
                vis_l3 = show_cam_on_image(
                    rgb_img, grayscale_cam_l3, use_rgb=True)

                # ⭐ 執行 Layer 4 RSA 內部權重視覺化 (放棄容易受雜訊干擾的 Grad-CAM)
                # 將 14x14 的注意力圖插值放大到 512x512
                attn_map = F.interpolate(spatial_attn, size=(
                    512, 512), mode='bilinear', align_corners=False)
                attn_map = attn_map.squeeze().cpu().numpy()

                # 歸一化權重到 0~1 之間以便繪製熱力圖
                attn_map = (attn_map - attn_map.min()) / \
                    (attn_map.max() - attn_map.min() + 1e-8)
                vis_l4_rsa = show_cam_on_image(rgb_img, attn_map, use_rgb=True)

                # 繪製 1x3 並排對比圖
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                # 判斷是否預測正確以決定標題顏色
                color = 'green' if str(pred_class) == str(class_id) else 'red'
                fig.suptitle(f"True: {class_id} | Pred: {pred_class} (Conf: {pred_score*100:.1f}%)",
                             fontsize=16, color=color, fontweight='bold')

                # [子圖 1] 原始圖片
                axes[0].imshow(cropped_img)
                axes[0].set_title("Original Image", fontsize=14)
                axes[0].axis('off')

                # [子圖 2] Layer 3 注意力 (紋理與細節)
                axes[1].imshow(vis_l3)
                axes[1].set_title(
                    "Layer 3: Local Details (Grad-CAM)", fontsize=14)
                axes[1].axis('off')

                # [子圖 3] Layer 4 注意力 (真正的空間遮罩)
                axes[2].imshow(vis_l4_rsa)
                axes[2].set_title(
                    "Layer 4: Semantic Activation Map", fontsize=14)
                axes[2].axis('off')

                plt.tight_layout()

                # 儲存圖片
                base_filename = os.path.splitext(os.path.basename(img_path))[0]
                save_path = os.path.join(
                    class_save_dir, f"dual_cam_{base_filename}.png")
                plt.savefig(save_path)

                # 釋放記憶體，避免大量圖片導致 RAM / VRAM 爆炸
                plt.close(fig)
                del input_tensor, outputs, scaled_outputs, probabilities, spatial_attn, attn_map

            except Exception as e:
                print(f"\nError processing {img_path}: {e}")

    print(
        f"\nAll done! Visualizations are categorized by class in: {args.save_dir}")


if __name__ == '__main__':
    main()
