import torch
import os
import numpy as np
import argparse
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from dataset import ImageDataset
from model import ImageClassificationModel
from utils import plot_class_distribution, plot_per_class_error, plot_correlation_analysis, plot_long_tail_accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tta', type=str, default='none',
                        choices=['none', 'flip', 'rotational'])
    parser.add_argument('--model_path', type=str,
                        default='./Model_Weight/best_model.pth')
    parser.add_argument('--config_name', type=str, default='39th')
    parser.add_argument('--img_size', type=int, default=512)
    args = parser.parse_args()

    PLOT_SAVE_DIR = f"./Plot/{args.config_name}/{args.tta}_tta"
    os.makedirs(PLOT_SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_transform = transforms.Compose([
        transforms.Resize(600), transforms.CenterCrop(512),
        transforms.ToTensor(), transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_dataset = ImageDataset(
        root_dir="./Dataset/data", split="val", transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=16,
                            shuffle=False, num_workers=4)

    model = ImageClassificationModel(
        num_classes=100, pretrained=False).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()
    running_loss, correct_preds, total_preds = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Validating ({args.tta})", colour="cyan"):
            images, labels = images.to(device), labels.to(device)

            if args.tta == 'none':
                logits = model(images)
                probs = torch.softmax(logits * 30.0, dim=1)
            elif args.tta == 'flip':
                logits = model(images)
                probs = (torch.softmax(logits * 30.0, dim=1) +
                         torch.softmax(model(torch.flip(images, dims=[3])), dim=1)) / 2.0
            elif args.tta == 'rotational':
                logits = model(images)
                probs = (torch.softmax(logits * 30.0, dim=1) +
                         torch.softmax(model(torch.flip(images, dims=[3])), dim=1) +
                         torch.softmax(model(torch.rot90(images, k=1, dims=[2, 3])), dim=1) +
                         torch.softmax(model(torch.rot90(images, k=3, dims=[2, 3])), dim=1)) / 4.0

            loss = criterion(logits, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(probs, 1)
            correct_preds += torch.sum(preds == labels.data).item()
            total_preds += images.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(f"\n [{args.tta.upper()} TTA] Val Loss: {running_loss/total_preds:.4f} | Val Acc: {(correct_preds/total_preds)*100:.2f}%")

    train_labels = ImageDataset(
        root_dir="./Dataset/data", split="train", transform=None).targets
    train_counts = plot_class_distribution(
        data_dir="./Dataset/data/train", output_path=PLOT_SAVE_DIR)
    error_rates = plot_per_class_error(all_preds, all_labels, save_path=os.path.join(
        PLOT_SAVE_DIR, "per_class_error.png"))
    plot_long_tail_accuracy(train_labels, all_preds, all_labels,
                            save_path=os.path.join(PLOT_SAVE_DIR, "long_tail.png"))
    plot_correlation_analysis(train_counts, error_rates, output_path=os.path.join(
        PLOT_SAVE_DIR, "correlation.png"))


if __name__ == "__main__":
    main()
