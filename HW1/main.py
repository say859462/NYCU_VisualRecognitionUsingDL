from utils import (
    plot_training_curves,
    plot_per_class_error,
    plot_long_tail_accuracy,
)
from val import validate_one_epoch
from train import train_one_epoch
from model import ImageClassificationModel
from dataset import ImageDataset

import torch
import os
import math
import json
import time
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from torch.utils.data import DataLoader
from torchvision import transforms

cudnn.benchmark = True


def build_optimizer(model, lr_base):
    return optim.AdamW(
        model.get_parameter_groups(lr_base),
        weight_decay=5e-4
    )


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

        progress = (self.last_epoch - self.warmup_epochs) / max(
            1, self.T_max - self.warmup_epochs
        )
        progress = min(max(progress, 0.0), 1.0)

        return [
            self.eta_min + (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * progress)) / 2
            for base_lr in self.base_lrs
        ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config.json')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    batch_size = config['batch_size']
    num_epochs = config.get('num_epochs', 30)
    lr_base = config.get('learning_rate', 1e-4)
    early_stopping_patience = config.get('early_stopping_patience', 10)
    num_classes = config['num_classes']
    data_dir = config['data_dir']
    resume_training = config['resume_training']

    checkpoint_path = config['checkpoint_path']
    best_model_path = config['best_model_path']
    best_loss_model_path = config['best_loss_model_path']
    best_full_model_path = config.get(
        'best_full_model_path',
        './Model_Weight/best_full_stageA_model.pth'
    )

    stage_b_start_epoch = config.get('stage_b_start_epoch', 9)

    num_subcenters = config.get('num_subcenters', 3)
    embed_dim = config.get('embed_dim', 256)

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(448, scale=(0.55, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(
            degrees=15,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
            shear=5,
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
        transforms.Resize(512),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = ImageDataset(
        root_dir=data_dir, split="train", transform=train_transform
    )
    val_dataset = ImageDataset(
        root_dir=data_dir, split="val", transform=val_transform
    )

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
        embed_dim=embed_dim
    ).to(device)

    if not model.check_parameters():
        print("The number of parameter is greater than 100,000,000!")
        return

    criterion_train = nn.CrossEntropyLoss(label_smoothing=0.05).to(device)
    criterion_val = nn.CrossEntropyLoss().to(device)

    optimizer = build_optimizer(model, lr_base)
    scheduler = WarmUpCosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        warmup_epochs=5,
        eta_min=1e-6
    )

    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')

    start_epoch = 0
    epochs_no_improve = 0

    best_stage_a_acc = 0.0
    best_stage_a_loss = float('inf')

    best_stage_b_acc = 0.0
    best_stage_b_loss_for_acc = float('inf')
    best_stage_b_loss_only = float('inf')

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'full_val_acc': [],
        'local_val_acc': [],
        'fused_val_acc': []
    }
    best_val_preds, best_val_labels = [], []

    if resume_training and os.path.exists(checkpoint_path):
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

        best_stage_a_acc = checkpoint.get('best_stage_a_acc', 0.0)
        best_stage_a_loss = checkpoint.get('best_stage_a_loss', float('inf'))

        best_stage_b_acc = checkpoint.get('best_stage_b_acc', 0.0)
        best_stage_b_loss_for_acc = checkpoint.get(
            'best_stage_b_loss_for_acc', float('inf'))
        best_stage_b_loss_only = checkpoint.get(
            'best_stage_b_loss_only', float('inf'))

        history = checkpoint['history']
        epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
        best_val_preds = checkpoint.get('best_val_preds', [])
        best_val_labels = checkpoint.get('best_val_labels', [])

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(
            f"✅ Successfully loaded checkpoint! Resuming from Epoch {start_epoch + 1}."
        )

    training_start_time = time.time()

    try:
        for epoch in range(start_epoch, num_epochs):
            current_epoch = epoch + 1
            in_stage_b = current_epoch >= stage_b_start_epoch

            print(f"\n--- Epoch {current_epoch}/{num_epochs} ---")
            if not in_stage_b:
                print(
                    "Stage A | CE(full) + background suppression aux + "
                    "CLS Cross-Attention full/global backbone"
                )
            else:
                print(
                    "Stage B | CE(fused) + lambda_full*CE(full) + "
                    "lambda_local*CE(local) + cross-attention bbox local crop"
                )

            train_loss, train_acc = train_one_epoch(
                model=model,
                train_loader=train_loader,
                criterion=criterion_train,
                epoch=current_epoch,
                optimizer=optimizer,
                device=device,
                scaler=scaler,

                stage_b_start_epoch=config.get('stage_b_start_epoch', 9),

                stage_a_full_loss_weight=config.get(
                    'stage_a_full_loss_weight', 1.0),
                stage_a_bg_aux_weight=config.get(
                    'stage_a_bg_aux_weight', 0.15),
                stage_a_bg_threshold=config.get('stage_a_bg_threshold', 0.45),
                stage_a_bg_suppress_strength=config.get(
                    'stage_a_bg_suppress_strength', 0.35),
                stage_a_bg_blur_kernel=config.get('stage_a_bg_blur_kernel', 7),
                stage_a_proto_diversity_weight=config.get(
                    'stage_a_proto_diversity_weight', 0.0),

                stage_b_fused_loss_weight=config.get(
                    'stage_b_fused_loss_weight', 1.0),
                stage_b_full_loss_weight=config.get(
                    'stage_b_full_loss_weight', 0.5),
                stage_b_local_loss_weight=config.get(
                    'stage_b_local_loss_weight', 0.1),
                stage_b_proto_diversity_weight=config.get(
                    'stage_b_proto_diversity_weight', 0.0),

                local_crop_threshold=config.get('local_crop_threshold', 0.55),
                local_crop_padding_ratio=config.get(
                    'local_crop_padding_ratio', 0.12),
                local_min_crop_ratio=config.get('local_min_crop_ratio', 0.20),
                local_max_crop_ratio=config.get('local_max_crop_ratio', 0.75),
                local_fallback_crop_ratio=config.get(
                    'local_fallback_crop_ratio', 0.40),
            )

            val_loss, val_acc, val_preds, val_labels, metrics = validate_one_epoch(
                model, val_loader, criterion_val, device, config, current_epoch
            )

            scheduler.step()

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['full_val_acc'].append(metrics["full_acc"])
            history['local_val_acc'].append(
                0.0 if metrics["local_acc"] is None else metrics["local_acc"]
            )
            history['fused_val_acc'].append(
                0.0 if metrics["fused_acc"] is None else metrics["fused_acc"]
            )

            if not in_stage_b:
                print(
                    f"LR: {optimizer.param_groups[-1]['lr']:.6f} | "
                    f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                    f"Val Loss(full): {val_loss:.4f} | Val Acc(full): {metrics['full_acc']:.2f}%"
                )

                if metrics["full_acc"] > best_stage_a_acc or (
                    metrics["full_acc"] == best_stage_a_acc and val_loss < best_stage_a_loss
                ):
                    best_stage_a_acc = metrics["full_acc"]
                    best_stage_a_loss = val_loss
                    torch.save(model.state_dict(), best_full_model_path)
                    print(
                        f"🌟 Best Stage-A FULL model saved ({best_stage_a_acc:.2f}%)")

            else:
                print(
                    f"LR: {optimizer.param_groups[-1]['lr']:.6f} | "
                    f"Train Loss: {train_loss:.4f} | Train Acc(fused): {train_acc:.2f}% | "
                    f"Val Loss(fused): {val_loss:.4f} | "
                    f"Val Full: {metrics['full_acc']:.2f}% | "
                    f"Val Local: {metrics['local_acc']:.2f}% | "
                    f"Val Fused: {metrics['fused_acc']:.2f}%"
                )

                improved = False

                if metrics["fused_acc"] > best_stage_b_acc or (
                    metrics["fused_acc"] == best_stage_b_acc and val_loss < best_stage_b_loss_for_acc
                ):
                    best_stage_b_acc = metrics["fused_acc"]
                    best_stage_b_loss_for_acc = val_loss
                    best_val_preds = val_preds
                    best_val_labels = val_labels
                    improved = True

                    torch.save(model.state_dict(), best_model_path)
                    print(
                        f"🌟 Best Stage-B FUSED model saved ({best_stage_b_acc:.2f}%)")

                if val_loss < best_stage_b_loss_only:
                    best_stage_b_loss_only = val_loss
                    improved = True

                    torch.save(model.state_dict(), best_loss_model_path)
                    print(
                        f"💡 Best Stage-B LOSS model saved ({best_stage_b_loss_only:.4f})")

                if improved:
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    print(
                        f"No improvement! {epochs_no_improve}/{early_stopping_patience}"
                    )

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),

                'best_stage_a_acc': best_stage_a_acc,
                'best_stage_a_loss': best_stage_a_loss,

                'best_stage_b_acc': best_stage_b_acc,
                'best_stage_b_loss_for_acc': best_stage_b_loss_for_acc,
                'best_stage_b_loss_only': best_stage_b_loss_only,

                'history': history,
                'epochs_no_improve': epochs_no_improve,
                'best_val_preds': best_val_preds,
                'best_val_labels': best_val_labels
            }, checkpoint_path)

            if in_stage_b and epochs_no_improve >= early_stopping_patience:
                break

    except KeyboardInterrupt:
        print("\n" + "=" * 50 + "\nDetected Keyboard Interrupt.\n" + "=" * 50)

    hours, rem = divmod(time.time() - training_start_time, 3600)
    minutes, seconds = divmod(rem, 60)

    plot_dir = "./Plot"
    os.makedirs(plot_dir, exist_ok=True)

    if history['train_loss']:
        print("\n📈 Generating analysis plots...")
        plot_training_curves(
            history['train_loss'], history['val_loss'],
            history['train_acc'], history['val_acc'],
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
        f"\n✅ Training Completed. "
        f"Best Stage-A Full Acc: {best_stage_a_acc:.2f}% | "
        f"Best Stage-B Fused Acc: {best_stage_b_acc:.2f}% | "
        f"Total Time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
    )


if __name__ == "__main__":
    main()
