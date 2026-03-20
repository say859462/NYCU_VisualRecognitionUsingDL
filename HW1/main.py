from utils import plot_training_curves, plot_per_class_error, plot_long_tail_accuracy
from val import validate_one_epoch
from train import train_one_epoch
from model import ImageClassificationModel
from dataset import ImageDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import argparse
import json
import time
import numpy as np

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config.json')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    BATCH_SIZE = config['batch_size']
    NUM_EPOCHS = config['num_epochs']
    LR_BASE = config.get('learning_rate', 1e-4)
    EARLY_STOPPING_PATIENCE = config.get('early_stopping_patience', 10)
    NUM_CLASSES = config['num_classes']
    DATA_DIR = config['data_dir']
    
    CHECKPOINT_PATH = config['checkpoint_path']
    BEST_MODEL_PATH = config['best_model_path']

    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(512, scale=(0.3, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2))
    ])

    val_transform = transforms.Compose([
        transforms.Resize(576),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ImageDataset(root_dir=DATA_DIR, split="train", transform=train_transform)
    val_dataset = ImageDataset(root_dir=DATA_DIR, split="val", transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)

    model = ImageClassificationModel(num_classes=NUM_CLASSES, pretrained=True).to(device)

    # 凍結淺層特徵 (ImageNet的基礎幾何特徵)
    for name, param in model.named_parameters():
        if any(name.startswith(prefix) for prefix in ['conv1', 'bn1', 'layer1']):
            param.requires_grad = False
        else:
            param.requires_grad = True

    # ⭐ 恢復使用最乾淨的 CrossEntropyLoss
    criterion = nn.CrossEntropyLoss().to(device)

    head_params = []
    backbone_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'classifier' in name or 'embedding' in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    # 標準的差分學習率
    param_groups = [
        {'params': backbone_params, 'lr': LR_BASE * 0.1}, 
        {'params': head_params, 'lr': LR_BASE * 2.0},      
    ]

    optimizer = optim.AdamW(param_groups, weight_decay=5e-4)

    # 包含 Warmup 的 CosineAnnealing 排程器
    from torch.optim.lr_scheduler import CosineAnnealingLR
    class WarmUpCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
        def __init__(self, optimizer, T_max, warmup_epochs=5, eta_min=1e-6, last_epoch=-1):
            self.warmup_epochs = warmup_epochs
            self.cosine_scheduler = CosineAnnealingLR(optimizer, T_max=T_max - warmup_epochs, eta_min=eta_min)
            super().__init__(optimizer, last_epoch)

        def get_lr(self):
            if self.last_epoch < self.warmup_epochs:
                return [base_lr * ((self.last_epoch + 1) / self.warmup_epochs) for base_lr in self.base_lrs]
            return self.cosine_scheduler.get_lr()

        def step(self, epoch=None):
            if self.last_epoch >= self.warmup_epochs - 1:
                self.cosine_scheduler.step(epoch)
            super().step(epoch)

    scheduler = WarmUpCosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, warmup_epochs=5, eta_min=1e-6)

    if not model.check_parameters():
        return
    scaler = torch.amp.GradScaler('cuda')

    start_epoch = 0
    best_val_acc = 0.0
    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_preds, best_val_labels = [], []

    training_start_time = time.time()
    try:
        for epoch in range(start_epoch, NUM_EPOCHS):
            print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")

            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device, scaler, max_grad_norm=2.0
            )

            val_loss, val_acc, val_preds, val_labels = validate_one_epoch(
                model, val_loader, criterion, device
            )

            scheduler.step()
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            # ⭐ 注意索引已經修正為 1
            lr_backbone = optimizer.param_groups[0]['lr']
            lr_head = optimizer.param_groups[1]['lr']
            print(f"LR_Backbone: {lr_backbone:.6f} | LR_Head: {lr_head:.6f}")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

            if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
                best_val_acc = val_acc
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_val_preds, best_val_labels = val_preds, val_labels
                torch.save(model.state_dict(), BEST_MODEL_PATH)
                print(f"🌟 Found a better model! Updated {BEST_MODEL_PATH} ({best_val_acc:.2f}%)")
            else:
                epochs_no_improve += 1
                print(f"No improvement. Early Stopping counter: {epochs_no_improve}/{EARLY_STOPPING_PATIENCE}")

            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"\n Early stopping triggered! Stopping training.")
                break
    except KeyboardInterrupt:
        print("\n" + "="*50 + "\nDetected Keyboard Interrupt.\n" + "="*50)

    training_end_time = time.time()
    hours, rem = divmod(training_end_time - training_start_time, 3600)
    minutes, seconds = divmod(rem, 60)

    if len(history['train_loss']) > 0:
        plot_training_curves(history['train_loss'], history['val_loss'], history['train_acc'], history['val_acc'])
        if best_val_preds and best_val_labels:
            plot_per_class_error(best_val_preds, best_val_labels, num_classes=NUM_CLASSES, save_path="./Plot/error_dist.png")
            plot_long_tail_accuracy(train_labels=train_dataset.targets, val_preds=best_val_preds, val_labels=best_val_labels, num_classes=NUM_CLASSES, save_path="./Plot/long_tail_acc.png")
            print(" Plots saved to ./Plot/")

    print(f"\n Training Completed. Best Val Acc: {best_val_acc:.2f}% | Time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")

if __name__ == "__main__":
    main()