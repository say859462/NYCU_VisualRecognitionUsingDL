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

# 載入你寫好的模型
from model import ImageClassificationModel

# 載入 Grad-CAM 套件
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def main():
    parser = argparse.ArgumentParser(
        description="Batch Grad-CAM Visualization for ResNet152")
    parser.add_argument('--dir_path', type=str, default='./Dataset/data/val/2',
                        help='Path to the directory containing images (e.g., ./Dataset/data/val/2)')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of images to randomly sample')
    parser.add_argument('--model_path', type=str,
                        default='./Model_Weight/10th/best_model.pth', help='Path to model weight')
    parser.add_argument('--num_classes', type=int,
                        default=100, help='Number of classes')
    parser.add_argument('--save_dir', type=str,
                        default='./Plot/GradCAM_Outputs', help='Directory to save heatmaps')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 確保儲存目錄存在
    os.makedirs(args.save_dir, exist_ok=True)

    # 2. 獲取資料夾內所有圖片並進行隨機採樣
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    all_images = [
        os.path.join(args.dir_path, f) for f in os.listdir(args.dir_path)
        if f.lower().endswith(valid_extensions) and os.path.isfile(os.path.join(args.dir_path, f))
    ]

    if not all_images:
        print(f"Error: No images found in {args.dir_path}")
        return

    # 如果要求的樣本數大於實際圖片數，則取全圖
    actual_samples = min(args.num_samples, len(all_images))
    sampled_image_paths = random.sample(all_images, actual_samples)
    print(
        f"Found {len(all_images)} images. Randomly sampled {actual_samples} images for Grad-CAM.")

    # 3. 初始化模型並載入權重
    model = ImageClassificationModel(
        num_classes=args.num_classes, pretrained=False).to(device)
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Loaded model weight from: {args.model_path}")
    else:
        raise FileNotFoundError(f"Model weight not found at {args.model_path}")

    model.eval()

    # 4. 設定 Grad-CAM 目標層 (ResNet152 最後一層卷積)
    target_layers = [model.backbone[7][-1]]

    # target_layers = [model.cbam]

    # 影像前處理定義
    preprocess_geo = transforms.Compose([
        transforms.Resize(400),
        transforms.CenterCrop(384)
    ])
    preprocess_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    # 5. 初始化 Grad-CAM 物件 (移到迴圈外以節省資源)
    cam = GradCAM(model=model, target_layers=target_layers)

    # 6. 開始批量處理
    print("\nGenerating Heatmaps...")
    for img_path in tqdm(sampled_image_paths, desc="Processing Images"):
        try:
            raw_img = Image.open(img_path).convert('RGB')
            cropped_img = preprocess_geo(raw_img)
            input_tensor = preprocess_tensor(
                cropped_img).unsqueeze(0).to(device)
            rgb_img = np.float32(cropped_img) / 255.0

            # 執行 Grad-CAM
            grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
            visualization = show_cam_on_image(
                rgb_img, grayscale_cam, use_rgb=True)

            # 取得預測結果
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                pred_class = probabilities.argmax().item()
                pred_score = probabilities[pred_class].item()

            # 繪製並存檔
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(cropped_img)
            plt.title("Original (Cropped)", fontsize=14)
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(visualization)

            # 從檔名取得真實標籤 (假設資料夾名稱就是真實標籤，例如 ./Dataset/data/val/2 -> 真實標籤為 2)
            true_label = os.path.basename(os.path.dirname(img_path))
            color = 'green' if str(pred_class) == true_label else 'red'

            plt.title(f"Grad-CAM\nTrue: {true_label} | Pred: {pred_class} (Conf: {pred_score*100:.1f}%)",
                      fontsize=14, color=color)
            plt.axis('off')

            plt.tight_layout()

            # 建立輸出檔名
            base_filename = os.path.splitext(os.path.basename(img_path))[0]
            save_path = os.path.join(
                args.save_dir, f"heatmap_{true_label}_{base_filename}.png")
            plt.savefig(save_path)
            plt.close()  # 記得關閉 plt 以免記憶體外洩

        except Exception as e:
            print(f"\nError processing {img_path}: {e}")

    print(f"\nAll done! Heatmaps saved to: {args.save_dir}")


if __name__ == '__main__':
    main()
