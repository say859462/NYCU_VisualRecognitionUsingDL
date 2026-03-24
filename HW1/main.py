from utils import (
    plot_training_curves,
    plot_per_class_error,
    plot_long_tail_accuracy,
    BalancedSoftmaxLoss,
    load_hard_classes_from_csv,
    build_pair_aware_sample_weights
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

from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms

cudnn.benchmark = True


def get_stage(epoch, stage1_epochs):
    return 1 if epoch < stage1_epochs else 2


def build_optimizer(model, lr_base, stage):
    return optim.AdamW(
        model.get_parameter_groups(lr_base, stage),
        weight_decay=3e-4
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

        progress = (self.last_epoch - self.warmup_epochs) / max(1, self.T_max - self.warmup_epochs)
        progress = min(max(progress, 0.0), 1.0)
        return [
            self.eta_min + (base_lr - self.eta_min) *
            (1 + math.cos(math.pi * progress)) / 2
            for base_lr in self.base_lrs
        ]


def build_scheduler(optimizer, stage, stage1_epochs, total_epochs):
    if stage == 1:
        return WarmUpCosineAnnealingLR(
            optimizer,
            T_max=max(1, stage1_epochs),
            warmup_epochs=5,
            eta_min=1e-6
        )
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_epochs - stage1_epochs),
        eta_min=1e-6
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config.json')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    lr_base = config.get('learning_rate', 1e-4)
    early_stopping_patience = config.get('early_stopping_patience', 12)
    num_classes = config['num_classes']
    data_dir = config['data_dir']
    resume_training = config['resume_training']
    checkpoint_path = config['checkpoint_path']
    best_model_path = config['best_model_path']

    # 推薦設定：Stage1 18 epoch，Stage2 4 epoch（總共 22）
    stage1_epochs = config.get('stage1_epochs', 18)

    hard_pairs_csv = config.get(
        'hard_pairs_csv',
        './Plot/EXP52_Diagnostic/top_confused_pairs.csv'
    )
    hard_pairs_top_k = config.get('hard_pairs_top_k', 10)
    hard_pair_boost = config.get('hard_pair_boost', 1.8)

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(512, scale=(0.55, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(
            degrees=10,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
            shear=5
        ),
        transforms.ColorJitter(
            brightness=0.12,
            contrast=0.12,
            saturation=0.08,
            hue=0.02
        ),
        transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize(576),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = ImageDataset(root_dir=data_dir, split="train", transform=train_transform)
    val_dataset = ImageDataset(root_dir=data_dir, split="val", transform=val_transform)

    # Stage 1: hard-pair weighted sampling
    hard_classes = load_hard_classes_from_csv(hard_pairs_csv, top_k_pairs=hard_pairs_top_k)
    pair_weights = build_pair_aware_sample_weights(
        train_dataset.targets,
        hard_classes,
        boost=hard_pair_boost
    )
    stage1_sampler = WeightedRandomSampler(
        weights=pair_weights,
        num_samples=len(pair_weights),
        replacement=True
    )

    train_loader_stage1 = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=stage1_sampler,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )

    # Stage 2: natural distribution + shuffle
    train_loader_stage2 = DataLoader(
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

    model = ImageClassificationModel(num_classes=num_classes, pretrained=True).to(device)
    if not model.check_parameters():
        print("The number of parameter is greater than 100,000,000!")
        return

    class_counts = torch.bincount(
        torch.tensor(train_dataset.targets),
        minlength=num_classes
    ).float().to(device)

    criterion_stage1 = nn.CrossEntropyLoss(label_smoothing=0.05).to(device)
    criterion_stage2 = BalancedSoftmaxLoss(class_counts).to(device)
    criterion_val = nn.CrossEntropyLoss().to(device)

    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')

    start_epoch, best_val_acc, best_val_loss, epochs_no_improve = 0, 0.0, float('inf'), 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_preds, best_val_labels = [], []

    initial_stage = get_stage(start_epoch, stage1_epochs)
    model.set_train_stage(initial_stage)
    optimizer = build_optimizer(model, lr_base, initial_stage)
    scheduler = build_scheduler(optimizer, initial_stage, stage1_epochs, num_epochs)

    loaded_optimizer_state = False
    active_stage = initial_stage

    if resume_training and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint['best_val_acc']
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        history = checkpoint['history']
        epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
        best_val_preds = checkpoint.get('best_val_preds', [])
        best_val_labels = checkpoint.get('best_val_labels', [])

        resume_stage = get_stage(start_epoch, stage1_epochs)
        model.set_train_stage(resume_stage)
        optimizer = build_optimizer(model, lr_base, resume_stage)
        scheduler = build_scheduler(optimizer, resume_stage, stage1_epochs, num_epochs)

        checkpoint_stage = get_stage(checkpoint['epoch'], stage1_epochs)
        if checkpoint_stage == resume_stage:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            loaded_optimizer_state = True
            active_stage = resume_stage

        print(f"✅ Successfully loaded checkpoint! Resuming from Epoch {start_epoch+1}.")

    training_start_time = time.time()

    try:
        for epoch in range(start_epoch, num_epochs):
            stage = get_stage(epoch, stage1_epochs)

            if stage != active_stage:
                model.set_train_stage(stage)
                optimizer = build_optimizer(model, lr_base, stage)
                scheduler = build_scheduler(optimizer, stage, stage1_epochs, num_epochs)
                print(
                    f"\n🔄 Switching to Stage {stage}: "
                    f"{'hard-pair + full/crop CE' if stage == 1 else 'short classifier calibration + Balanced Softmax'}"
                )
                active_stage = stage

            loaded_optimizer_state = False
            criterion_train = criterion_stage1 if stage == 1 else criterion_stage2
            train_loader = train_loader_stage1 if stage == 1 else train_loader_stage2
            dual_view = (stage == 1)

            print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
            print(
                f"Stage {stage} | "
                f"{'hard-pair sampler + CE + dual-view crop' if stage == 1 else 'shuffle + Balanced Softmax'}"
            )

            train_loss, train_acc = train_one_epoch(
                model=model,
                train_loader=train_loader,
                criterion=criterion_train,
                epoch=epoch + 1,
                optimizer=optimizer,
                device=device,
                scaler=scaler,
                stage=stage,
                dual_view=dual_view,
                full_weight=0.7,
                crop_weight=0.3
            )

            val_loss, val_acc, val_preds, val_labels = validate_one_epoch(
                model, val_loader, criterion_val, device
            )

            scheduler.step()
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            print(
                f"LR: {optimizer.param_groups[-1]['lr']:.6f} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
            )

            if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
                best_val_acc, best_val_loss, epochs_no_improve = val_acc, val_loss, 0
                best_val_preds, best_val_labels = val_preds, val_labels
                torch.save(model.state_dict(), best_model_path)
                print(f"🌟 Found a better model! Updated {best_model_path} ({best_val_acc:.2f}%)")
            else:
                epochs_no_improve += 1
                print(f"No improvement. Early Stopping counter: {epochs_no_improve}/{early_stopping_patience}")

            torch.save({
                'epoch': epoch,
                'stage': stage,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'best_val_loss': best_val_loss,
                'history': history,
                'epochs_no_improve': epochs_no_improve,
                'best_val_preds': best_val_preds,
                'best_val_labels': best_val_labels
            }, checkpoint_path)

            if epochs_no_improve >= early_stopping_patience:
                break

    except KeyboardInterrupt:
        print("\n" + "="*50 + "\nDetected Keyboard Interrupt.\n" + "="*50)

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
        f"\n✅ Training Completed. Best Val Acc: {best_val_acc:.2f}% | "
        f"Total Time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
    )


if __name__ == "__main__":
    main()