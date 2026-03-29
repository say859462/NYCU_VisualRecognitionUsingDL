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

from dataset import ImageDataset
from model import ImageClassificationModel
from train import train_one_epoch
from utils import plot_long_tail_accuracy, plot_per_class_error, plot_training_curves
from val import validate_one_epoch

cudnn.benchmark = True


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

        progress = (self.last_epoch - self.warmup_epochs) / max(1, self.T_max - self.warmup_epochs)
        progress = min(max(progress, 0.0), 1.0)
        return [
            self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * progress)) / 2
            for base_lr in self.base_lrs
        ]


def build_optimizer(model, lr_base):
    return optim.AdamW(model.get_parameter_groups(lr_base), weight_decay=5e-4)


def get_train_geometry(epoch, config):
    stage1_epochs = config.get("pmg_stage1_epochs", 4)
    stage2_epochs = config.get("pmg_stage2_epochs", 4)
    if epoch <= stage1_epochs + stage2_epochs:
        return {
            "resize": config.get("curriculum_stage12_resize", 576),
            "crop": config.get("curriculum_stage12_crop", 512),
            "tag": "stage12_geometry",
        }
    return {
        "resize": config.get("curriculum_stage3_resize", 576),
        "crop": config.get("curriculum_stage3_crop", 512),
        "tag": "stage3_geometry",
    }


def build_train_transform(resize_size, crop_size):
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop((crop_size, crop_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.02)
        ], p=0.40),
        transforms.RandomApply([
            transforms.RandomRotation(degrees=10)
        ], p=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_eval_transform(eval_resize):
    return transforms.Compose([
        transforms.Resize((eval_resize, eval_resize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_loader(data_dir, split, batch_size, transform, num_workers=8, shuffle=False):
    dataset = ImageDataset(root_dir=data_dir, split=split, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return dataset, loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config.json")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as file:
        config = json.load(file)

    batch_size = config["batch_size"]
    num_epochs = config.get("num_epochs", 30)
    lr_base = config.get("learning_rate", 1e-4)
    early_stopping_patience = config.get("early_stopping_patience", 12)
    data_dir = config["data_dir"]
    checkpoint_path = config["checkpoint_path"]
    best_model_path = config["best_model_path"]
    best_loss_model_path = config["best_loss_model_path"]
    eval_resize = config.get("eval_resize", 576)
    resume_training = config.get("resume_training", False)

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_transform = build_eval_transform(eval_resize)
    _, val_loader = build_loader(
        data_dir=data_dir,
        split="val",
        batch_size=batch_size,
        transform=val_transform,
        num_workers=config.get("num_workers", 8),
        shuffle=False,
    )

    model = ImageClassificationModel(
        num_classes=config["num_classes"],
        pretrained=True,
        num_subcenters=config.get("num_subcenters", 3),
        embed_dim=config.get("embed_dim", 256),
        backbone_name=config.get("backbone_name", "resnet152_partial_res2net"),
    ).to(device)

    if not model.check_parameters():
        print("The number of parameters is greater than 100,000,000.")
        return

    criterion_train = nn.CrossEntropyLoss(label_smoothing=float(config.get("label_smoothing", 0.05))).to(device)
    criterion_val = nn.CrossEntropyLoss().to(device)
    optimizer = build_optimizer(model, lr_base)
    scheduler = WarmUpCosineAnnealingLR(
        optimizer,
        T_max=max(1, num_epochs),
        warmup_epochs=config.get("warmup_epochs", 5),
        eta_min=1e-6,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    start_epoch = 0
    epochs_no_improve = 0
    best_val_acc = 0.0
    best_val_loss_for_acc = float("inf")
    best_val_loss_only = float("inf")
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "train_concat_acc": [],
        "val_concat_acc": [],
    }
    best_val_preds = []
    best_val_labels = []

    if resume_training and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_acc = checkpoint.get("best_val_acc", 0.0)
        best_val_loss_for_acc = checkpoint.get("best_val_loss_for_acc", float("inf"))
        best_val_loss_only = checkpoint.get("best_val_loss_only", float("inf"))
        history = checkpoint.get("history", history)
        epochs_no_improve = checkpoint.get("epochs_no_improve", 0)
        best_val_preds = checkpoint.get("best_val_preds", [])
        best_val_labels = checkpoint.get("best_val_labels", [])
        print(f"Resumed training from epoch {start_epoch + 1}")

    train_dataset = None
    train_loader = None
    current_loader_tag = None
    training_start_time = time.time()

    try:
        for epoch_idx in range(start_epoch, num_epochs):
            epoch = epoch_idx + 1
            print(f"\n--- Epoch {epoch}/{num_epochs} ---")
            geometry = get_train_geometry(epoch, config)

            if current_loader_tag != geometry["tag"]:
                train_transform = build_train_transform(geometry["resize"], geometry["crop"])
                train_dataset, train_loader = build_loader(
                    data_dir=data_dir,
                    split="train",
                    batch_size=batch_size,
                    transform=train_transform,
                    num_workers=config.get("num_workers", 8),
                    shuffle=True,
                )
                current_loader_tag = geometry["tag"]
                print(
                    f"Switched train geometry -> Resize({geometry['resize']}) + RandomCrop({geometry['crop']})"
                )

            train_stats = train_one_epoch(
                model=model,
                train_loader=train_loader,
                criterion=criterion_train,
                epoch=epoch,
                optimizer=optimizer,
                device=device,
                scaler=scaler,
                config=config,
            )
            val_stats = validate_one_epoch(
                model=model,
                val_loader=val_loader,
                criterion=criterion_val,
                device=device,
                config=config,
                epoch=epoch,
            )
            scheduler.step()

            history["train_loss"].append(train_stats["loss"])
            history["train_acc"].append(train_stats["main_acc"])
            history["val_loss"].append(val_stats["loss"])
            history["val_acc"].append(val_stats["main_acc"])
            history["train_concat_acc"].append(train_stats["concat_acc"])
            history["val_concat_acc"].append(val_stats["concat_acc"])

            print(train_stats["stage_cfg"]["stage_name"])
            print(
                f"Train geometry -> Resize({geometry['resize']}) + RandomCrop({geometry['crop']}) | Eval resize -> {eval_resize}"
            )
            print(
                f"Loss weights -> global: {train_stats['stage_cfg']['global_weight']:.2f}, "
                f"part2: {train_stats['stage_cfg']['part2_weight']:.2f}, "
                f"part4: {train_stats['stage_cfg']['part4_weight']:.2f}, "
                f"concat: {train_stats['stage_cfg']['concat_weight']:.2f}, "
                f"crop_cls: {train_stats['stage_cfg']['crop_cls_weight']:.2f}, "
                f"consistency: {train_stats['stage_cfg']['consistency_weight']:.2f}"
            )
            print(
                f"LR: {scheduler.get_last_lr()[-1]:.6f} | "
                f"Train Loss: {train_stats['loss']:.4f} | Train Main Acc: {train_stats['main_acc']:.2f}% | "
                f"Train Concat Acc: {train_stats['concat_acc']:.2f}% | Train Crop Acc: {train_stats['crop_concat_acc']:.2f}% | "
                f"Val Loss: {val_stats['loss']:.4f} | Val Main Acc: {val_stats['main_acc']:.2f}% | "
                f"Val Concat Acc: {val_stats['concat_acc']:.2f}%"
            )

            improved = False
            if (val_stats["main_acc"] > best_val_acc) or (
                val_stats["main_acc"] == best_val_acc and val_stats["loss"] < best_val_loss_for_acc
            ):
                best_val_acc = val_stats["main_acc"]
                best_val_loss_for_acc = val_stats["loss"]
                best_val_preds = val_stats["preds"]
                best_val_labels = val_stats["labels"]
                torch.save(model.state_dict(), best_model_path)
                print(f"Best model saved ({best_val_acc:.2f}%)")
                epochs_no_improve = 0
                improved = True
            else:
                epochs_no_improve += 1
                print(f"No improvement! {epochs_no_improve}/{early_stopping_patience}")

            if val_stats["loss"] < best_val_loss_only:
                best_val_loss_only = val_stats["loss"]
                torch.save(model.state_dict(), best_loss_model_path)
                print(f"Best loss model saved ({best_val_loss_only:.4f})")

            torch.save(
                {
                    "epoch": epoch_idx,
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
                },
                checkpoint_path,
            )

            if epochs_no_improve >= early_stopping_patience:
                print("Early stopping triggered.")
                break

        print(f"Training finished in {(time.time() - training_start_time) / 60.0:.2f} minutes")

        plot_training_curves(
            history["train_loss"],
            history["val_loss"],
            history["train_acc"],
            history["val_acc"],
            save_path=config.get("training_curve_path", "./Plot/training_curves_pairwise_crop.png"),
        )
        if best_val_preds and best_val_labels:
            plot_per_class_error(
                best_val_preds,
                best_val_labels,
                num_classes=config["num_classes"],
                save_path=config.get("error_curve_path", "./Plot/error_dist_pairwise_crop.png"),
            )
            corr = plot_long_tail_accuracy(
                os.path.join(data_dir, "train"),
                best_val_preds,
                best_val_labels,
                num_classes=config["num_classes"],
                save_path=config.get("long_tail_curve_path", "./Plot/long_tail_pairwise_crop.png"),
            )
            print(f"Long-tail correlation (train count vs val acc): {corr:.4f}")

    finally:
        del train_dataset, train_loader, val_loader
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
