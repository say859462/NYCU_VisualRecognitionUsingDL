from utils import plot_training_curves, plot_per_class_error, plot_long_tail_accuracy, ClassBalancedFocalLoss, get_cb_weights, RandomDiscreteRotation
from val import validate_one_epoch
from train import train_one_epoch
from model import ImageClassificationModel
from dataset import ImageDataset
import torch
import os
import json
import time
import numpy as np
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader
from torchvision import transforms
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
    RESUME_TRAINING = config['resume_training']
    CHECKPOINT_PATH = config['checkpoint_path']
    BEST_MODEL_PATH = config['best_model_path']

    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ⭐ 移除銳利化，加入 RandomErasing 來抹除「大拇指」等作弊特徵
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(512, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        RandomDiscreteRotation(angles=[0, 90, 270]),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.3, 3.3))
    ])

    val_transform = transforms.Compose([
        transforms.Resize(576), transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    train_dataset = ImageDataset(
        root_dir=DATA_DIR, split="train", transform=train_transform)
    val_dataset = ImageDataset(
        root_dir=DATA_DIR, split="val", transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=8, pin_memory=True, persistent_workers=True)

    model = ImageClassificationModel(
        num_classes=NUM_CLASSES, pretrained=True).to(device)

    head_params, backbone_params = [], []
    for name, param in model.named_parameters():
        if any(name.startswith(prefix) for prefix in ['stem', 'layer1']):
            param.requires_grad = False
        else:
            param.requires_grad = True
            if 'classifier' in name or 'bottleneck' in name:
                head_params.append(param)
            else:
                backbone_params.append(param)

    # ⭐ 降回標準 1e-4 的 Weight Decay
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': LR_BASE * 0.1},
        {'params': head_params, 'lr': LR_BASE * 2.0},
    ], weight_decay=1e-4)

    train_labels = train_dataset.targets
    cb_weights = get_cb_weights(
        train_labels, NUM_CLASSES, beta=0.999).to(device)

    # ⭐ 採用更溫和且有效的 Class-Balanced Focal Loss
    criterion_train = ClassBalancedFocalLoss(
        cb_weights=cb_weights, gamma=2.0).to(device)
    criterion_val = nn.CrossEntropyLoss().to(device)

    from torch.optim.lr_scheduler import CosineAnnealingLR

    class WarmUpCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
        def __init__(self, optimizer, T_max, warmup_epochs=5, eta_min=1e-6, last_epoch=-1):
            self.warmup_epochs = warmup_epochs
            self.cosine_scheduler = CosineAnnealingLR(
                optimizer, T_max=T_max - warmup_epochs, eta_min=eta_min)
            super().__init__(optimizer, last_epoch)

        def get_lr(self):
            if self.last_epoch < self.warmup_epochs:
                return [base_lr * ((self.last_epoch + 1) / self.warmup_epochs) for base_lr in self.base_lrs]
            return self.cosine_scheduler.get_lr()

        def step(self, epoch=None):
            if self.last_epoch >= self.warmup_epochs - 1:
                self.cosine_scheduler.step(epoch)
            super().step(epoch)

    scheduler = WarmUpCosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, warmup_epochs=5, eta_min=1e-6)

    if not model.check_parameters():
        return
    scaler = torch.amp.GradScaler('cuda')

    start_epoch, best_val_acc, best_val_loss, epochs_no_improve = 0, 0.0, float(
        'inf'), 0
    history = {'train_loss': [], 'val_loss': [],
               'train_acc': [], 'val_acc': []}
    best_val_preds, best_val_labels = [], []

    if RESUME_TRAINING and os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(
            CHECKPOINT_PATH, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint['best_val_acc']
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        history, epochs_no_improve = checkpoint['history'], checkpoint.get(
            'epochs_no_improve', 0)
        best_val_preds, best_val_labels = checkpoint.get(
            'best_val_preds', []), checkpoint.get('best_val_labels', [])
        print(
            f"✅ Successfully loaded checkpoint! Resuming from Epoch {start_epoch+1}.")

    training_start_time = time.time()
    try:
        for epoch in range(start_epoch, NUM_EPOCHS):
            print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")

            # 傳入 focal loss
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion_train, epoch+1, optimizer, device, scaler, max_grad_norm=2.0
            )
            # 驗證時使用 CE Loss 反映真實機率誤差
            val_loss, val_acc, val_preds, val_labels = validate_one_epoch(
                model, val_loader, criterion_val, device
            )

            scheduler.step()
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            print(
                f"LR_Backbone: {optimizer.param_groups[0]['lr']:.6f} | LR_Head: {optimizer.param_groups[1]['lr']:.6f}")
            print(
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

            if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
                best_val_acc, best_val_loss, epochs_no_improve = val_acc, val_loss, 0
                best_val_preds, best_val_labels = val_preds, val_labels
                torch.save(model.state_dict(), BEST_MODEL_PATH)
                print(
                    f"🌟 Found a better model! Updated {BEST_MODEL_PATH} ({best_val_acc:.2f}%)")
            else:
                epochs_no_improve += 1
                print(
                    f"No improvement. Early Stopping counter: {epochs_no_improve}/{EARLY_STOPPING_PATIENCE}")

            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(), 'best_val_acc': best_val_acc, 'best_val_loss': best_val_loss,
                'history': history, 'epochs_no_improve': epochs_no_improve, 'best_val_preds': best_val_preds, 'best_val_labels': best_val_labels,
            }, CHECKPOINT_PATH)

            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                break
    except KeyboardInterrupt:
        print("\n" + "="*50 + "\nDetected Keyboard Interrupt.\n" + "="*50)

    hours, rem = divmod(time.time() - training_start_time, 3600)
    minutes, seconds = divmod(rem, 60)

    if len(history['train_loss']) > 0:
        plot_training_curves(
            history['train_loss'], history['val_loss'], history['train_acc'], history['val_acc'])
        if best_val_preds and best_val_labels:
            plot_per_class_error(best_val_preds, best_val_labels,
                                 num_classes=NUM_CLASSES, save_path="./Plot/error_dist.png")
            plot_long_tail_accuracy(train_labels=train_dataset.targets, val_preds=best_val_preds,
                                    val_labels=best_val_labels, num_classes=NUM_CLASSES, save_path="./Plot/long_tail_acc.png")

    print(
        f"\n Training Completed. Best Val Acc: {best_val_acc:.2f}% | Time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")


if __name__ == "__main__":
    main()
