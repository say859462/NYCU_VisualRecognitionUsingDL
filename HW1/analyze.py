import torch
import os
import numpy as np
import argparse
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm

from dataset import ImageDataset
from model import ImageClassificationModel
from utils import (
    plot_class_distribution,
    plot_per_class_error,
    plot_correlation_analysis,
    plot_long_tail_accuracy,
    ClassBalancedFocalLoss
)


def main():
    parser = argparse.ArgumentParser(
        description="Model Analysis with TTA Options")
    parser.add_argument('--tta', type=str, default='none', choices=['none', 'flip', 'rotational'],
                        help='TTA mode: none, flip (Horizontal), rotational (4-Crop)')
    parser.add_argument('--model_path', type=str, default='./Model_Weight/best_model.pth',
                        help='Path to the model weights')

    parser.add_argument('--config_name', type=str, default='22th',
                        help='Name for the output directory')

    parser.add_argument('--img_size', type=int, default=512,
                        help='Crop size for inference')
    args = parser.parse_args()

    # Parameters
    DATA_DIR = "./Dataset/data"
    NUM_CLASSES = 100
    BATCH_SIZE = 16
    PLOT_SAVE_DIR = f"./Plot/{args.config_name}/{args.tta}_tta"
    os.makedirs(PLOT_SAVE_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Device: {device} | TTA Mode: {args.tta} | Image Size: {args.img_size}")

    # 推斷用的 Transform (可以根據需求調整 Resize 大小以榨取效能)
    val_transform = transforms.Compose([
        transforms.Resize(int(args.img_size * 1.15)),  # 保持比例縮放
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    val_dataset = ImageDataset(
        root_dir=DATA_DIR, split="val", transform=val_transform)
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print("Loading class distribution...")
    train_dataset = ImageDataset(
        root_dir=DATA_DIR, split="train", transform=None)
    train_labels = train_dataset.targets

    model = ImageClassificationModel(
        num_classes=NUM_CLASSES, pretrained=False).to(device)
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Loaded weights from: {args.model_path}")
    else:
        print(f"Error: Model not found at {args.model_path}")
        return

    # 初始化 Loss (用於分析)
    criterion = torch.nn.CrossEntropyLoss()

    model.eval()
    running_loss, correct_preds, total_preds = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Validating ({args.tta})", colour="cyan")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            # --- TTA 核心邏輯 ---
            if args.tta == 'none':
                outputs = model(images) * 20.0
                probs = torch.softmax(outputs, dim=1)

            elif args.tta == 'flip':
                outputs = model(images) * 20.0
                out_flip = model(torch.flip(images, dims=[3])) * 20.0
                probs = (torch.softmax(outputs, dim=1) +
                         torch.softmax(out_flip, dim=1)) / 2.0

            elif args.tta == 'rotational':
                outputs = model(images) * 20.0
                out_flip = model(torch.flip(images, dims=[3])) * 20.0
                out_rot90 = model(torch.rot90(images, k=1, dims=[2, 3])) * 20.0
                out_rot270 = model(torch.rot90(
                    images, k=3, dims=[2, 3])) * 20.0

                probs = (torch.softmax(outputs, dim=1) +
                         torch.softmax(out_flip, dim=1) +
                         torch.softmax(out_rot90, dim=1) +
                         torch.softmax(out_rot270, dim=1)) / 4.0

            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            _, preds = torch.max(probs, 1)
            correct_preds += torch.sum(preds == labels.data).item()
            total_preds += images.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = running_loss / total_preds
    val_acc = (correct_preds / total_preds) * 100

    print(
        f"\n [{args.tta.upper()} TTA Result] Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    # --- 繪圖分析 ---
    train_path = os.path.join(DATA_DIR, "train")
    train_counts = plot_class_distribution(
        data_dir=train_path, title="Train Set Statistics", output_path=PLOT_SAVE_DIR)
    error_rates = plot_per_class_error(all_preds, all_labels, num_classes=NUM_CLASSES, save_path=os.path.join(
        PLOT_SAVE_DIR, "per_class_error.png"))
    plot_long_tail_accuracy(train_labels=train_labels, val_preds=all_preds,
                            val_labels=all_labels, save_path=os.path.join(PLOT_SAVE_DIR, "long_tail.png"))

    if train_counts and error_rates:
        plot_correlation_analysis(train_counts, error_rates, output_path=os.path.join(
            PLOT_SAVE_DIR, "correlation.png"))


if __name__ == "__main__":
    main()
