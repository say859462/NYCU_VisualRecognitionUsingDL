import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import ImageDataset
from model import ImageClassificationModel

from val import validate_one_epoch
from utils import (
    plot_class_distribution,
    plot_per_class_error,
    plot_correlation_analysis,
    plot_long_tail_accuracy,
    ClassBalancedFocalLoss
)


def main():
    # Parameters
    DATA_DIR = "./Dataset/data"
    MODEL_PATH = "./Model_Weight/14th/best_model.pth"  # model weight
    NUM_CLASSES = 100
    BATCH_SIZE = 32
    PLOT_SAVE_DIR = "./Plot/14th"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    val_transform = transforms.Compose([
        transforms.Resize(640),
        transforms.CenterCrop(576),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    val_dataset = ImageDataset(
        root_dir=DATA_DIR, split="val", transform=val_transform)
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print("Loading train dataset to get class distribution for long-tail plot...")
    train_dataset = ImageDataset(
        root_dir=DATA_DIR, split="train", transform=None)
    train_labels = train_dataset.targets

    model = ImageClassificationModel(
        num_classes=NUM_CLASSES, pretrained=False).to(device)

    beta = 0.999
    class_sample_count = np.bincount(train_labels, minlength=NUM_CLASSES)

    effective_num = 1.0 - np.power(beta, class_sample_count)
    effective_num = np.maximum(effective_num, 1e-8)  # avoid division by zero
    cb_weights = (1.0 - beta) / np.array(effective_num)
    cb_weights = cb_weights / np.sum(cb_weights) * NUM_CLASSES

    class_weights = torch.FloatTensor(cb_weights).to(device)
    criterion = ClassBalancedFocalLoss(
        cb_weights=class_weights, gamma=2.0, label_smoothing=0.1)

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Load the model from : {MODEL_PATH}")
    else:
        print(
            f"Error: Couldn't find the model weight at {MODEL_PATH}. Please ensure the path is correct and the file exists.")
        return

    print("\nValidating the best model on validation set to get predictions and targets for analysis...")

    val_loss, val_acc, all_preds, all_labels = validate_one_epoch(
        model, val_loader, criterion, device
    )

    print("\n" + "="*50)
    print(
        f"🎉 [High-Res Inference Result] Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    print("="*50 + "\n")

    print("\nGenerating analysis plots...")

    train_path = os.path.join(DATA_DIR, "train")
    train_counts = plot_class_distribution(
        data_dir=train_path, title="Train Set Statistics")

    error_save_path = os.path.join(PLOT_SAVE_DIR, "val_per_class_error.png")
    error_rates = plot_per_class_error(
        all_preds, all_labels, num_classes=NUM_CLASSES, save_path=error_save_path
    )

    long_tail_save_path = os.path.join(PLOT_SAVE_DIR, "long_tail_acc.png")
    plot_long_tail_accuracy(
        train_labels=train_labels,
        val_preds=all_preds,
        val_labels=all_labels,
        num_classes=NUM_CLASSES,
        save_path=long_tail_save_path
    )
    print(f"Long-tail accuracy plot saved to {long_tail_save_path}")

    # Correlation analysis between training sample count and error rates
    if train_counts and error_rates:
        corr_save_path = os.path.join(
            PLOT_SAVE_DIR, "correlation_analysis.png")
        plot_correlation_analysis(
            train_counts, error_rates, output_path=corr_save_path)
        print(
            f"Correlation analysis completed. Plot saved to: {corr_save_path}")

    print(f"\nAnalyze completed. Plots saved to {PLOT_SAVE_DIR}")


if __name__ == "__main__":
    main()
