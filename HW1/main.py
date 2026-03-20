from utils import SimilarityLDAMLoss, plot_training_curves, plot_per_class_error, plot_long_tail_accuracy
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

    # ==============================================================================
    # ⭐ STAGE 2 參數覆寫區
    # ==============================================================================
    NUM_EPOCHS = 20          # 只需要微調 20 輪
    LR_BASE = 5e-5           # 極低的基底學習率
    # ==============================================================================

    EARLY_STOPPING_PATIENCE = config.get('early_stopping_patience', 10)
    NUM_CLASSES = config['num_classes']
    DATA_DIR = config['data_dir']

    # 將儲存檔名加上 _stage2 綴詞，避免覆蓋 Stage1 的神級權重
    CHECKPOINT_PATH = config['checkpoint_path'].replace('.pth', '_stage2.pth')
    BEST_MODEL_PATH = config['best_model_path']
    STAGE2_BEST_PATH = BEST_MODEL_PATH.replace('.pth', '_stage2.pth')

    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=8, pin_memory=True, persistent_workers=True)

    model = ImageClassificationModel(
        num_classes=100, pretrained=True).to(device)

    # ==============================================================================
    # ⭐ 載入 Stage 1 特徵 (含形狀不匹配防呆處理)
    # ==============================================================================
    if os.path.exists(BEST_MODEL_PATH):
        print(f"\n🚀 Loading Stage 1 features from {BEST_MODEL_PATH}...")
        checkpoint = torch.load(
            BEST_MODEL_PATH, map_location=device, weights_only=False)
        # 過濾掉舊分類器的權重，避免 Linear 與 NormedLinear 發生維度衝突
        filtered_checkpoint = {
            k: v for k, v in checkpoint.items() if 'classifier' not in k}
        model.load_state_dict(filtered_checkpoint, strict=False)
        print("✅ Successfully loaded backbone and embedding features!\n")
    else:
        print("⚠️ Warning: Stage 1 weights not found! Training from scratch.")

    for name, param in model.named_parameters():
        if any(name.startswith(prefix) for prefix in ['conv1', 'bn1', 'layer1']):
            param.requires_grad = False
        else:
            param.requires_grad = True

    train_labels = train_dataset.targets
    class_sample_count = np.bincount(train_labels, minlength=NUM_CLASSES)

    # ⭐ 換回 LDAM Loss 來對抗長尾分佈
    criterion = SimilarityLDAMLoss(
        cls_num_list=class_sample_count, max_m=0.45, s=20.0, alpha=0.1, keep_ratio=0.7
    ).to(device)
    val_criterion = nn.CrossEntropyLoss().to(device)

    head_params = []
    backbone_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'classifier' in name or 'embedding' in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    # ⭐ 極端不對稱學習率 (保護特徵，專心微調頭部)
    param_groups = [
        # Backbone 幾乎不動 (lr = 2.5e-6)
        {'params': backbone_params, 'lr': LR_BASE * 0.05},
        # Head 重點微調 (lr = 2.5e-4)
        {'params': head_params, 'lr': LR_BASE * 5.0},
    ]

    optimizer = optim.AdamW(param_groups, weight_decay=5e-4)

    # Stage 2 不需 warmup，直接平滑下降
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    if not model.check_parameters():
        return
    scaler = torch.amp.GradScaler('cuda')

    start_epoch = 0
    best_val_acc = 0.0
    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [],
               'train_acc': [], 'val_acc': []}
    best_val_preds, best_val_labels = [], []

    training_start_time = time.time()
    try:
        for epoch in range(start_epoch, NUM_EPOCHS):
            print(f"\n--- Stage 2 | Epoch {epoch+1}/{NUM_EPOCHS} ---")

            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device, scaler, max_grad_norm=2.0
            )

            val_loss, val_acc, val_preds, val_labels = validate_one_epoch(
                model, val_loader, val_criterion, device, s=20.0
            )

            scheduler.step()
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            lr_backbone = optimizer.param_groups[0]['lr']
            lr_head = optimizer.param_groups[1]['lr']
            print(f"LR_Backbone: {lr_backbone:.6f} | LR_Head: {lr_head:.6f}")
            print(
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

            if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
                best_val_acc = val_acc
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_val_preds, best_val_labels = val_preds, val_labels
                torch.save(model.state_dict(), STAGE2_BEST_PATH)
                print(
                    f"🌟 Found a better model! Updated {STAGE2_BEST_PATH} ({best_val_acc:.2f}%)")
            else:
                epochs_no_improve += 1
                print(
                    f"No improvement. Early Stopping counter: {epochs_no_improve}/{EARLY_STOPPING_PATIENCE}")

            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"\n Early stopping triggered! Stopping training.")
                break
    except KeyboardInterrupt:
        print("\n" + "="*50 + "\nDetected Keyboard Interrupt.\n" + "="*50)

    training_end_time = time.time()
    hours, rem = divmod(training_end_time - training_start_time, 3600)
    minutes, seconds = divmod(rem, 60)

    if len(history['train_loss']) > 0:
        plot_training_curves(
            history['train_loss'], history['val_loss'], history['train_acc'], history['val_acc'])
        if best_val_preds and best_val_labels:
            plot_per_class_error(best_val_preds, best_val_labels,
                                 num_classes=NUM_CLASSES, save_path="./Plot/stage2_error_dist.png")
            plot_long_tail_accuracy(train_labels=train_dataset.targets, val_preds=best_val_preds,
                                    val_labels=best_val_labels, num_classes=NUM_CLASSES, save_path="./Plot/stage2_long_tail_acc.png")
            print(" Plots saved to ./Plot/")

    print(
        f"\n Stage 2 Completed. Best Val Acc: {best_val_acc:.2f}% | Time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")


if __name__ == "__main__":
    main()
