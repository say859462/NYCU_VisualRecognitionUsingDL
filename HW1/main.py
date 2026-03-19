from utils import SimilarityLDAMLoss, SupConLoss, plot_training_curves, plot_per_class_error, plot_long_tail_accuracy
from val import validate_one_epoch
from train import train_one_epoch
from model import ImageClassificationModel
from dataset import ImageDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
import os
import argparse
import json
import time
import numpy as np

# ==============================================================================
# 0. Hardware Optimization
# ==============================================================================
import torch.backends.cudnn as cudnn
cudnn.benchmark = True


def main():
    # ==============================================================================
    # 1. Configuration & Arguments
    # ==============================================================================
    parser = argparse.ArgumentParser(
        description="Image Classification Model Training")
    parser.add_argument('--config', type=str,
                        default='./config.json', help='Path to config')
    args = parser.parse_args()

    print(f"Loading configuration file: {args.config}")
    with open(args.config, 'r') as f:
        config = json.load(f)

    BATCH_SIZE = config['batch_size']
    NUM_EPOCHS = config['num_epochs']
    LR_BASE = config.get('learning_rate', 2e-4)  # ⭐ 預訓練基底好，建議稍微調高基礎 LR
    EARLY_STOPPING_PATIENCE = config.get('early_stopping_patience', 40)
    NUM_CLASSES = config['num_classes']
    DATA_DIR = config['data_dir']
    CHECKPOINT_PATH = config['checkpoint_path']
    BEST_MODEL_PATH = config['best_model_path']
    RESUME_TRAINING = config['resume_training']

    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(BEST_MODEL_PATH), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ==============================================================================
    # 2. Data Preprocessing & Loaders
    # ==============================================================================
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(512, scale=(0.3, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2))
    ])

    val_transform = transforms.Compose([
        transforms.Resize(576),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    train_dataset = ImageDataset(
        root_dir=DATA_DIR, split="train", transform=train_transform)
    val_dataset = ImageDataset(
        root_dir=DATA_DIR, split="val", transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=8, pin_memory=True, persistent_workers=True)

    # ==============================================================================
    # 3. Model, Loss, Optimizer & Scheduler
    # ==============================================================================
    model = ImageClassificationModel(
        num_classes=100, pretrained=True).to(device)

    # ⭐ 修正 1：對齊官方 Backbone 名稱來凍結淺層 (選配，凍結 stem 與 layer1 可穩定初期訓練)
    for name, param in model.named_parameters():
        if any(name.startswith(prefix) for prefix in ['conv1', 'bn1', 'layer1']):
            param.requires_grad = False
        else:
            param.requires_grad = True

    train_labels = train_dataset.targets
    class_sample_count = np.bincount(train_labels, minlength=NUM_CLASSES)

    # 既然有預訓練好基底，直接啟用 LDAM + OHEM
    criterion = SimilarityLDAMLoss(
        cls_num_list=class_sample_count, max_m=0.45, s=20.0, alpha=0.1, keep_ratio=0.7
    ).to(device)

    supcon_loss_cbp = SupConLoss(temperature=0.1).to(device)
    supcon_loss_gem = SupConLoss(temperature=0.1).to(device)

    # ⭐ 修正 2：對齊新的模組名稱進行非對稱學習率分組
    head_params = []
    attention_cbp_params = []
    backbone_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'classifier' in name or 'embedding' in name:
            head_params.append(param)
        elif 'se_' in name or 'rsa_' in name or 'reduce' in name or 'cbp' in name or 'gem' in name:
            attention_cbp_params.append(param)
        else:
            backbone_params.append(param)  # 官方 resnext 預訓練權重

    param_groups = [
        {'params': backbone_params, 'lr': LR_BASE * 0.1},      # 預訓練權重只做微調
        {'params': attention_cbp_params, 'lr': LR_BASE * 1.0},  # 新加的注意力模組正常學習
        {'params': head_params, 'lr': LR_BASE * 2.0},          # 分類頭快速學習
    ]

    optimizer = optim.AdamW(param_groups, weight_decay=1e-3)

    from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
    warmup_epochs = 5
    cosine_epochs = NUM_EPOCHS - warmup_epochs
    warmup_sch = LinearLR(optimizer, start_factor=0.1,
                          total_iters=warmup_epochs)
    cosine_sch = CosineAnnealingLR(
        optimizer, T_max=cosine_epochs, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[
                             warmup_sch, cosine_sch], milestones=[warmup_epochs])

    if not model.check_parameters():
        return
    scaler = torch.amp.GradScaler('cuda')

    # ==============================================================================
    # 4. State Management
    # ==============================================================================
    start_epoch = 0
    best_val_acc = 0.0
    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [],
               'train_acc': [], 'val_acc': []}
    best_val_preds, best_val_labels = [], []

    if RESUME_TRAINING and os.path.exists(CHECKPOINT_PATH):
        print("Resume training is enabled and checkpoint found.")
        checkpoint = torch.load(
            CHECKPOINT_PATH, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint['best_val_acc']
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        history = checkpoint['history']

        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
        print(
            f" Successfully loaded checkpoint! Resuming from Epoch {start_epoch+1}.")
    else:
        print("\n Starting training from scratch.")

    # ==============================================================================
    # 5. Main Training Loop
    # ==============================================================================
    training_start_time = time.time()
    CROP_EPOCH = 10
    try:
        for epoch in range(start_epoch, NUM_EPOCHS):
            print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")

            # ⭐ 修正 3：取消階段性啟動，從第一輪直接開啟 Attention Crop 狂暴模式
            enable_crop = (epoch >= CROP_EPOCH)
            if enable_crop and epoch == CROP_EPOCH:
                print(f" [Attention Crop] activated from Epoch 1!")

            # 修正 4：正確傳遞 supcon_loss 給 train_one_epoch，並移除用不到的 center_loss
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device, scaler,
                supcon_loss_cbp=supcon_loss_cbp,
                supcon_loss_gem=supcon_loss_gem,
                max_grad_norm=2.0,
                use_crop=enable_crop
            )

            val_loss, val_acc, val_preds, val_labels = validate_one_epoch(
                model, val_loader, torch.nn.CrossEntropyLoss(), device)

            scheduler.step()
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            lr_backbone = optimizer.param_groups[0]['lr']
            lr_head = optimizer.param_groups[2]['lr']
            print(f"LR_Backbone: {lr_backbone:.6f} | LR_Head: {lr_head:.6f}")
            print(
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

            if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
                best_val_acc = val_acc
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_val_preds, best_val_labels = val_preds, val_labels
                torch.save(model.state_dict(), BEST_MODEL_PATH)
                print(
                    f"Found a better model. Updated {BEST_MODEL_PATH} ({best_val_acc:.2f}%)")
            else:
                epochs_no_improve += 1
                print(
                    f"No improvement. Early Stopping counter: {epochs_no_improve}/{EARLY_STOPPING_PATIENCE}")

            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_val_acc': best_val_acc,
                'best_val_loss': best_val_loss,
                'history': history,
                'epochs_no_improve': epochs_no_improve,
            }
            torch.save(checkpoint_data, CHECKPOINT_PATH)

            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"\n Early stopping triggered! Stopping training.")
                break
    except KeyboardInterrupt:
        print("\n" + "="*50 + "\nDetected Keyboard Interrupt.\n" + "="*50)

    # ==============================================================================
    # 6. Post-Training Operations
    # ==============================================================================
    training_end_time = time.time()
    hours, rem = divmod(training_end_time - training_start_time, 3600)
    minutes, seconds = divmod(rem, 60)

    if len(history['train_loss']) > 0:
        plot_training_curves(
            history['train_loss'], history['val_loss'], history['train_acc'], history['val_acc'])
        if best_val_preds and best_val_labels:
            plot_per_class_error(best_val_preds, best_val_labels,
                                 num_classes=NUM_CLASSES, save_path="./Plot/error_dist.png")
            plot_long_tail_accuracy(train_labels=train_dataset.targets, val_preds=best_val_preds,
                                    val_labels=best_val_labels, num_classes=NUM_CLASSES, save_path="./Plot/long_tail_acc.png")
            print(" Plots saved to ./Plot/")

    print(
        f"\n Training completed. Best Val Acc: {best_val_acc:.2f}% | Time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")


if __name__ == "__main__":
    main()
