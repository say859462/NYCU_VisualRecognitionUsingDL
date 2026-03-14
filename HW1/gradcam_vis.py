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
        description="Batch Grad-CAM Visualization for All Classes")
    # 將路徑改為整個 val 資料夾的根目錄
    parser.add_argument('--val_dir', type=str, default='./Dataset/data/val',
                        help='Path to the validation dataset root directory')
    # 新增參數：每個類別要抽樣畫幾張圖 (預設 3 張)
    parser.add_argument('--num_samples_per_class', type=int, default=3,
                        help='Number of images to randomly sample per class')
    parser.add_argument('--model_path', type=str,
                        default='./Model_Weight/best_model.pth', help='Path to model weight')
    parser.add_argument('--num_classes', type=int,
                        default=100, help='Number of classes')
    parser.add_argument('--save_dir', type=str,
                        default='./Plot/GradCAM_Outputs/11th', help='Directory to save heatmaps')
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

    # 建議優先觀察 cbam_l3，看模型是否捕捉到蝴蝶顏色或小目標細節
    target_layers = [model.cbam_l3]
    # 如果想看高階語義判斷（如青蛙問題），可以改用：
    # target_layers = [model.cbam_l4]

    # 影像前處理定義 (回歸 CenterCrop 以對齊第11次實驗)
    preprocess_geo = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(448)
    ])
    preprocess_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    # 初始化 Grad-CAM 物件
    cam = GradCAM(model=model, target_layers=target_layers)

    # 4. 開始遍歷所有類別資料夾
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

    print(f"\nProcessing all {args.num_classes} classes...")
    # 外層迴圈：跑 0 ~ 99 類
    for class_id in tqdm(range(args.num_classes), desc="Classes Processed"):
        class_dir = os.path.join(args.val_dir, str(class_id))

        if not os.path.exists(class_dir):
            continue

        # 建立該類別的專屬輸出子資料夾
        class_save_dir = os.path.join(args.save_dir, str(class_id))
        os.makedirs(class_save_dir, exist_ok=True)

        # 獲取該類別所有圖片
        all_images = [
            os.path.join(class_dir, f) for f in os.listdir(class_dir)
            if f.lower().endswith(valid_extensions) and os.path.isfile(os.path.join(class_dir, f))
        ]

        if not all_images:
            continue

        # 隨機採樣 (如果該類別圖片太少，就全拿)
        actual_samples = min(args.num_samples_per_class, len(all_images))
        sampled_image_paths = random.sample(all_images, actual_samples)

        # 內層迴圈：處理該類別被抽中的圖片
        for img_path in sampled_image_paths:
            try:
                raw_img = Image.open(img_path).convert('RGB')
                cropped_img = preprocess_geo(raw_img)
                input_tensor = preprocess_tensor(
                    cropped_img).unsqueeze(0).to(device)
                rgb_img = np.float32(cropped_img) / 255.0

                # 執行 Grad-CAM
                grayscale_cam = cam(
                    input_tensor=input_tensor, targets=None)[0, :]
                visualization = show_cam_on_image(
                    rgb_img, grayscale_cam, use_rgb=True)

                # 取得預測結果
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)[
                        0]
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

                # 判斷是否預測正確以決定標題顏色
                color = 'green' if str(pred_class) == str(class_id) else 'red'
                plt.title(f"Grad-CAM\nTrue: {class_id} | Pred: {pred_class} (Conf: {pred_score*100:.1f}%)",
                          fontsize=14, color=color)
                plt.axis('off')

                plt.tight_layout()

                # 儲存到專屬類別資料夾中
                base_filename = os.path.splitext(os.path.basename(img_path))[0]
                save_path = os.path.join(
                    class_save_dir, f"heatmap_{base_filename}.png")
                plt.savefig(save_path)
                plt.close()

            except Exception as e:
                print(f"\nError processing {img_path}: {e}")

    print(f"\nAll done! Heatmaps are categorized by class in: {args.save_dir}")


if __name__ == '__main__':
    main()
