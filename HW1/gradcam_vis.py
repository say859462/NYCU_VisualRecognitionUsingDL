import torch
import os
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from model import ImageClassificationModel
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_dir', type=str, default='./Dataset/data/val')
    parser.add_argument('--num_samples_per_class', type=int, default=3)
    parser.add_argument('--model_path', type=str,
                        default='./Model_Weight/best_model.pth')
    parser.add_argument('--save_dir', type=str,
                        default='./Plot/GradCAM_Outputs/42th')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    model = ImageClassificationModel(
        num_classes=100, pretrained=False).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # ⭐ 因為 model 現在只回傳 logits，不需要 Wrapper 了！直接綁定！
    cam = GradCAM(model=model, target_layers=[model.layer4[-1]])

    preprocess_geo = transforms.Compose(
        [transforms.Resize(600), transforms.CenterCrop(512)])
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
                # ⭐ 修正：乘上 30.0 讓 Softmax 能正確運作
                probs = torch.nn.functional.softmax(logits * 30.0, dim=1)[0]
                pred_class, pred_score = probs.argmax().item(), probs.max().item()

            grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
            vis = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            color = 'green' if str(pred_class) == str(class_id) else 'red'
            fig.suptitle(
                f"True: {class_id} | Pred: {pred_class} (Conf: {pred_score*100:.1f}%)", color=color, fontweight='bold')

            axes[0].imshow(cropped_img)
            axes[0].axis('off')
            axes[0].set_title("Original")
            axes[1].imshow(vis)
            axes[1].axis('off')
            axes[1].set_title("Res2Net Grad-CAM")

            plt.savefig(os.path.join(
                class_save_dir, f"cam_{os.path.splitext(os.path.basename(img_path))[0]}.png"))
            plt.close(fig)


if __name__ == '__main__':
    main()
