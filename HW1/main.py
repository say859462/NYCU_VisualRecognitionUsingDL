from utils import (
    plot_training_curves,
    plot_per_class_error,
    plot_long_tail_accuracy,
)
from val import validate_one_epoch
from train import train_one_epoch
from model import ImageClassificationModel
from dataset import ImageDataset

import argparse
import json
import math
import os
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

cudnn.benchmark = True


def build_optimizer(model, lr_base):
    return optim.AdamW(model.get_parameter_groups(lr_base), weight_decay=5e-4)


class WarmUpCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_max, warmup_epochs=5, eta_min=1e-6, last_epoch=-1):
        self.T_max = max(1, T_max)
        self.warmup_epochs = min(warmup_epochs, max(0, self.T_max - 1))
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.warmup_epochs > 0 and self.last_epoch < self.warmup_epochs:
            scale = (self.last_epoch + 1) / self.warmup_epochs
            return [base_lr * scale for base_lr in self.base_lrs]

        if self.T_max == self.warmup_epochs:
            return [self.eta_min for _ in self.base_lrs]

        progress = (self.last_epoch - self.warmup_epochs) / \
            max(1, self.T_max - self.warmup_epochs)
        progress = min(max(progress, 0.0), 1.0)

        return [
            self.eta_min + (base_lr - self.eta_min) *
            (1 + math.cos(math.pi * progress)) / 2
            for base_lr in self.base_lrs
        ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config.json")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    batch_size = config["batch_size"]
    num_epochs = config.get("num_epochs", 30)
    lr_base = config.get("learning_rate", 1e-4)
    early_stopping_patience = config.get("early_stopping_patience", 10)
    num_classes = config["num_classes"]
    data_dir = config["data_dir"]
    resume_training = config.get("resume_training", False)

    checkpoint_path = config["checkpoint_path"]
    best_model_path = config["best_model_path"]
    best_loss_model_path = config["best_loss_model_path"]

    num_subcenters = config.get("num_subcenters", 3)
    embed_dim = config.get("embed_dim", 256)

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = transforms.Compose([
        transforms.Resize((576, 576)),
        transforms.RandomCrop(512),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(
            degrees=7,
            translate=(0.02, 0.02),
            scale=(0.98, 1.02),
            shear=2,
        ),
        transforms.ColorJitter(
            brightness=0.12,
            contrast=0.12,
            saturation=0.08,
            hue=0.02,
        ),
        transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = ImageDataset(
        root_dir=data_dir, split="train", transform=train_transform)
    val_dataset = ImageDataset(
        root_dir=data_dir, split="val", transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )

    model = ImageClassificationModel(
        num_classes=num_classes,
        pretrained=True,
        num_subcenters=num_subcenters,
        embed_dim=embed_dim,
    ).to(device)

    if not model.check_parameters():
        print("The number of parameters is greater than 100,000,000!")
        return

    criterion_train = nn.CrossEntropyLoss(label_smoothing=0.05).to(device)
    criterion_val = nn.CrossEntropyLoss().to(device)

    optimizer = build_optimizer(model, lr_base)
    scheduler = WarmUpCosineAnnealingLR(
        optimizer, T_max=max(1, num_epochs), warmup_epochs=5, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    start_epoch = 0
    epochs_no_improve = 0
    best_val_acc = 0.0
    best_val_loss_for_acc = float("inf")
    best_val_loss_only = float("inf")
    history = {"train_loss": [], "train_acc": [],
               "val_loss": [], "val_acc": []}
    best_val_preds = []
    best_val_labels = []

    if resume_training and os.path.exists(checkpoint_path):
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_acc = checkpoint.get("best_val_acc", 0.0)
        best_val_loss_for_acc = checkpoint.get(
            "best_val_loss_for_acc", float("inf"))
        best_val_loss_only = checkpoint.get("best_val_loss_only", float("inf"))
        history = checkpoint.get("history", history)
        epochs_no_improve = checkpoint.get("epochs_no_improve", 0)
        best_val_preds = checkpoint.get("best_val_preds", [])
        best_val_labels = checkpoint.get("best_val_labels", [])
        print(f"Resumed training from epoch {start_epoch + 1}")

    training_start_time = time.time()

    try:
        for epoch in range(start_epoch, num_epochs):
            current_epoch = epoch + 1
            print(f"\n--- Epoch {current_epoch}/{num_epochs} ---")

            train_loss, train_acc, stage_cfg = train_one_epoch(
                model=model,
                train_loader=train_loader,
                criterion=criterion_train,
                epoch=current_epoch,
                optimizer=optimizer,
                device=device,
                scaler=scaler,
                config=config,
            )

            val_loss, val_acc, val_preds, val_labels, _ = validate_one_epoch(
                model=model,
                val_loader=val_loader,
                criterion=criterion_val,
                device=device,
                config=config,
                epoch=current_epoch,
            )

            scheduler.step()

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            print(stage_cfg["stage_name"])
            print(
                "Loss weights -> "
                f"global: {stage_cfg['global_weight']:.2f}, "
                f"part2: {stage_cfg['part2_weight']:.2f}, "
                f"part4: {stage_cfg['part4_weight']:.2f}, "
                f"concat: {stage_cfg['concat_weight']:.2f}"
            )

            print(
                f"LR: {optimizer.param_groups[-1]['lr']:.6f} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
            )

            improved = False

            if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss_for_acc):
                best_val_acc = val_acc
                best_val_loss_for_acc = val_loss
                best_val_preds = val_preds
                best_val_labels = val_labels
                improved = True
                torch.save(model.state_dict(), best_model_path)
                print(f"🌟 Best model saved ({best_val_acc:.2f}%)")

            if val_loss < best_val_loss_only:
                best_val_loss_only = val_loss
                improved = True
                torch.save(model.state_dict(), best_loss_model_path)
                print(f"💡 Best loss model saved ({best_val_loss_only:.4f})")

            if improved:
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                print(
                    f"No improvement! {epochs_no_improve}/{early_stopping_patience}")

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_acc": best_val_acc,
                "best_val_loss_for_acc": best_val_loss_for_acc,
                "best_val_loss_only": best_val_loss_only,
                "history": history,
                "epochs_no_improve": epochs_no_improve,
                "best_val_preds": best_val_preds,
                "best_val_labels": best_val_labels,
            }, checkpoint_path)

            if epochs_no_improve >= early_stopping_patience:
                break

    except KeyboardInterrupt:
        print("\n" + "=" * 50 + "\nDetected Keyboard Interrupt.\n" + "=" * 50)

    hours, rem = divmod(time.time() - training_start_time, 3600)
    minutes, seconds = divmod(rem, 60)

    plot_dir = "./Plot"
    os.makedirs(plot_dir, exist_ok=True)

    if history["train_loss"]:
        print("\n📈 Generating analysis plots...")
        plot_training_curves(
            history["train_loss"], history["val_loss"],
            history["train_acc"], history["val_acc"],
            save_path=os.path.join(plot_dir, "training_curves.png")
        )

        if best_val_preds and best_val_labels:
            plot_per_class_error(
                best_val_preds,
                best_val_labels,
                num_classes=num_classes,
                save_path=os.path.join(plot_dir, "error_dist.png")
            )
            plot_long_tail_accuracy(
                train_labels=train_dataset.targets,
                val_preds=best_val_preds,
                val_labels=best_val_labels,
                num_classes=num_classes,
                save_path=os.path.join(plot_dir, "long_tail_acc.png")
            )

    print(
        f"\n✅ Training Completed. Best Val Acc: {best_val_acc:.2f}% | "
        f"Total Time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
    )


if __name__ == "__main__":
    main()
