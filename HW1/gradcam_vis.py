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

from model import ImageClassificationModel
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def main():
    # ... (前面的 argparse 參數設定保持不變) ...
    parser = argparse.ArgumentParser(
        description="Pure ResNeXt Grad-CAM Visualization")
    parser.add_argument('--val_dir', type=str, default='./Dataset/data/val')
    parser.add_argument('--num_samples_per_class', type=int, default=3)
    parser.add_argument('--model_path', type=str,
                        default='./Model_Weight/best_model.pth')
    parser.add_argument('--num_classes', type=int, default=100)
    parser.add_argument('--save_dir', type=str,
                        default='./Plot/GradCAM_Outputs/36th')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    model = ImageClassificationModel(
        num_classes=args.num_classes, pretrained=False).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # ⭐ 修正：綁定官方 ResNeXt 的 Layer 3 與 Layer 4 的最後一個 Block
    cam_l3 = GradCAM(model=model, target_layers=[model.layer3[-1]])
    cam_l4 = GradCAM(model=model, target_layers=[model.layer4[-1]])

    preprocess_geo = transforms.Compose(
        [transforms.Resize(576), transforms.CenterCrop(512)])
    preprocess_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

    for class_id in tqdm(range(args.num_classes), desc="Classes Processed"):
        class_dir = os.path.join(args.val_dir, str(class_id))
        if not os.path.exists(class_dir):
            continue

        class_save_dir = os.path.join(args.save_dir, str(class_id))
        os.makedirs(class_save_dir, exist_ok=True)

        all_images = [os.path.join(class_dir, f) for f in os.listdir(
            class_dir) if f.lower().endswith(valid_extensions)]
        sampled_image_paths = random.sample(all_images, min(
            args.num_samples_per_class, len(all_images)))

        for img_path in sampled_image_paths:
            raw_img = Image.open(img_path).convert('RGB')
            cropped_img = preprocess_geo(raw_img)
            input_tensor = preprocess_tensor(
                cropped_img).unsqueeze(0).to(device)
            rgb_img = np.float32(cropped_img) / 255.0

            with torch.no_grad():
                # ⭐ 修正：現在只會回傳 logits
                outputs, _ = model(input_tensor)
                probabilities = torch.nn.functional.softmax(
                    outputs * 15.0, dim=1)[0]
                pred_class = probabilities.argmax().item()
                pred_score = probabilities[pred_class].item()

            # 產生兩層的熱力圖
            grayscale_cam_l3 = cam_l3(
                input_tensor=input_tensor, targets=None)[0, :]
            vis_l3 = show_cam_on_image(rgb_img, grayscale_cam_l3, use_rgb=True)

            grayscale_cam_l4 = cam_l4(
                input_tensor=input_tensor, targets=None)[0, :]
            vis_l4 = show_cam_on_image(rgb_img, grayscale_cam_l4, use_rgb=True)

            # 繪製圖片
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            color = 'green' if str(pred_class) == str(class_id) else 'red'
            fig.suptitle(
                f"True: {class_id} | Pred: {pred_class} (Conf: {pred_score*100:.1f}%)", color=color, fontweight='bold')

            axes[0].imshow(cropped_img)
            axes[0].set_title("Original Image")
            axes[0].axis('off')
            axes[1].imshow(vis_l3)
            axes[1].set_title("Layer 3: Grad-CAM")
            axes[1].axis('off')
            axes[2].imshow(vis_l4)
            axes[2].set_title("Layer 4: Grad-CAM")
            axes[2].axis('off')

            base_filename = os.path.splitext(os.path.basename(img_path))[0]
            plt.savefig(os.path.join(class_save_dir,
                        f"cam_{base_filename}.png"))
            plt.close(fig)


if __name__ == '__main__':
    main()
