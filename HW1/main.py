from utils import plot_training_curves, plot_per_class_error, plot_long_tail_accuracy, get_cb_weights, SupConLoss, CurricularFace
from val import validate_one_epoch
from train import train_one_epoch
from model import ImageClassificationModel
from dataset import ImageDataset
import torch
import os
import math
import json
import time
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

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(448, scale=(0.3, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.RandomRotation(15),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(500),
        transforms.CenterCrop(448),
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

    if not model.check_parameters():
        print("The number of parameter is greater than 100,000,00!")
        return

    # 差分學習率 (解凍 Layer 2 必須用極小學習率)
    backbone_l2 = [p for n, p in model.named_parameters(
    ) if 'layer2' in n and p.requires_grad]
    backbone_l3_l4 = [p for n, p in model.named_parameters() if (
        'layer3' in n or 'layer4' in n) and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if not any(
        x in n for x in ['stem', 'layer1', 'layer2', 'layer3', 'layer4']) and p.requires_grad]

    optimizer = optim.AdamW([
        {'params': backbone_l2, 'lr': LR_BASE * 0.1},
        {'params': backbone_l3_l4, 'lr': LR_BASE * 1.0},
        {'params': head_params, 'lr': LR_BASE * 1.5},
    ], weight_decay=3e-4)

    cb_weights = get_cb_weights(
        train_dataset.targets, num_classes=NUM_CLASSES).to(device)
    criterion_supcon = SupConLoss(temperature=0.07).to(device)
    criterion_val = nn.CrossEntropyLoss().to(device)

    class WarmUpCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
        def __init__(self, optimizer, T_max, warmup_epochs=5, eta_min=1e-5, last_epoch=-1):
            self.warmup_epochs = warmup_epochs
            self.T_max = T_max
            self.eta_min = eta_min
            super().__init__(optimizer, last_epoch)

        def get_lr(self):
            if self.last_epoch < self.warmup_epochs:
                return [base_lr * ((self.last_epoch + 1) / self.warmup_epochs) for base_lr in self.base_lrs]
            progress = (self.last_epoch - self.warmup_epochs) / \
                (self.T_max - self.warmup_epochs)
            return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * progress)) / 2 for base_lr in self.base_lrs]

    scheduler = WarmUpCosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, warmup_epochs=5, eta_min=1e-6)

    scaler = torch.amp.GradScaler('cuda')

    start_epoch, best_val_acc, best_val_loss, epochs_no_improve = 0, 0.0, float(
        'inf'), 0
    history = {'train_loss': [], 'val_loss': [],
               'train_acc': [], 'val_acc': []}
    best_val_preds, best_val_labels = [], []

    STAGE_1B_START = 15  # 進入 Epoch 16
    CRT_START_EPOCH = 30  # 進入 Epoch 31
    criterion_curricular = CurricularFace(s=30.0, m=0.5).to(device)
    # ⭐ 補回 Checkpoint 讀取邏輯
    # ⭐ 修正後的 Checkpoint 讀取邏輯
    if RESUME_TRAINING and os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(
            CHECKPOINT_PATH, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

        start_epoch = checkpoint['epoch'] + 1
        if 'criterion_curricular_state_dict' in checkpoint:
            criterion_curricular.load_state_dict(
                checkpoint['criterion_curricular_state_dict'])
        print(
            f"✅ Successfully loaded checkpoint! Resuming from Epoch {start_epoch+1}.")
        if start_epoch > CRT_START_EPOCH:
            print(f"🚀 Detected Resume during Stage 2 (cRT). Re-configuring optimizer...")
            model.freeze_features_for_crt()  # 先凍結特徵

            # ⭐ 修正：必須與 Line 161 的設定完全一致，否則參數群組對不起來
            head_params, transformer_params = model.get_classifier_parameters()
            optimizer = optim.AdamW([
                {'params': head_params, 'lr': 1e-4},
                {'params': transformer_params, 'lr': 1e-5}  # 極小學習率微調
            ], weight_decay=1e-4)

            # cRT 階段使用的 scheduler
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=NUM_EPOCHS-CRT_START_EPOCH, eta_min=1e-6)

        # 現在參數群組數量一致了，可以安全載入
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        best_val_acc = checkpoint['best_val_acc']
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        history = checkpoint['history']
        epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
        best_val_preds = checkpoint.get('best_val_preds', [])
        best_val_labels = checkpoint.get('best_val_labels', [])
        print(
            f"✅ Successfully loaded checkpoint! Resuming from Epoch {start_epoch+1}.")

    training_start_time = time.time()
    try:
        for epoch in range(start_epoch, NUM_EPOCHS):
            print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")

            if epoch < STAGE_1B_START:
                # 🥇 Stage 1A: 基礎建設
                model.dropblock.drop_prob = 0.0  # 確保關閉
                criterions = {
                    'ce': nn.CrossEntropyLoss(label_smoothing=0.1).to(device),
                    'supcon': criterion_supcon, 'use_supcon': True,
                    'do_saliency_cutmix': False,
                    'pmg_weights': (1.5, 1.0, 0.5)
                }
            elif epoch < CRT_START_EPOCH:
                # 🥈 Stage 1B: 全局鍛鍊 (CutMix ON, DropBlock 必須 OFF 避免衝突)
                model.dropblock.drop_prob = 0.0  # ⭐ 解決衝突：徹底關閉 DropBlock
                criterions = {
                    'ce': nn.CrossEntropyLoss(label_smoothing=0.1).to(device),
                    'supcon': None, 'use_supcon': False,
                    'do_saliency_cutmix': True,  # 依賴 Transformer 提供穩定的注意力圖
                    'pmg_weights': (0.5, 1.0, 1.5)
                }
            else:
                # 🥉 Stage 2: 度量純化
                model.dropblock.drop_prob = 0.0
                if epoch == CRT_START_EPOCH:
                    print("\n🚀 [GEAR SHIFT] 啟動微調版 cRT + CurricularFace！")
                    model.freeze_features_for_crt()

                    # ⭐ 取得兩組參數
                    head_params, transformer_params = model.get_classifier_parameters()

                    # ⭐ 差分學習率：分類頭正常訓練，Transformer 極微幅調整
                    optimizer = optim.AdamW([
                        {'params': head_params, 'lr': 1e-4},
                        {'params': transformer_params, 'lr': 1e-5}  # 僅用 1/10 的學習率
                    ], weight_decay=1e-4)

                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=NUM_EPOCHS-CRT_START_EPOCH, eta_min=1e-6)

                criterions = {
                    'ce': criterion_curricular,  # ⭐ 直接使用全域的實例
                    'supcon': None, 'use_supcon': False,
                    'do_saliency_cutmix': False,
                    'pmg_weights': (0.0, 0.0, 1.0)
                }

            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterions, epoch+1, optimizer, device, scaler)
            val_loss, val_acc, val_preds, val_labels = validate_one_epoch(
                model, val_loader, criterion_val, device)

            scheduler.step()
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            print(
                f"LR: {optimizer.param_groups[-1]['lr']:.6f} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

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

            # ⭐ 補回 Checkpoint 儲存邏輯 (每個 Epoch 結束都存檔)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'best_val_loss': best_val_loss,
                'history': history,
                'epochs_no_improve': epochs_no_improve,
                'best_val_preds': best_val_preds,
                'best_val_labels': best_val_labels,
                'criterion_curricular_state_dict': criterion_curricular.state_dict()
            }, CHECKPOINT_PATH)

            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                break
    except KeyboardInterrupt:
        print("\n" + "="*50 + "\nDetected Keyboard Interrupt.\n" + "="*50)

    # 🕒 訓練結束，計算總耗時
    hours, rem = divmod(time.time() - training_start_time, 3600)
    minutes, seconds = divmod(rem, 60)

    # 📊 繪製並保存分析圖表
    PLOT_DIR = "./Plot"
    os.makedirs(PLOT_DIR, exist_ok=True)

    if len(history['train_loss']) > 0:
        print("\n📈 Generating analysis plots...")
        plot_training_curves(
            history['train_loss'], history['val_loss'],
            history['train_acc'], history['val_acc'],
            save_path=os.path.join(PLOT_DIR, "training_curves.png")
        )

        if best_val_preds and best_val_labels:
            plot_per_class_error(
                best_val_preds, best_val_labels,
                num_classes=NUM_CLASSES,
                save_path=os.path.join(PLOT_DIR, "error_dist.png")
            )
            plot_long_tail_accuracy(
                train_labels=train_dataset.targets,
                val_preds=best_val_preds,
                val_labels=best_val_labels,
                num_classes=NUM_CLASSES,
                save_path=os.path.join(PLOT_DIR, "long_tail_acc.png")
            )

    print(
        f"\n✅ Training Completed. Best Val Acc: {best_val_acc:.2f}% | Total Time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")


if __name__ == "__main__":
    main()
