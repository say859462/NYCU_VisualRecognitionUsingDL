from utils import plot_training_curves, plot_per_class_error, plot_long_tail_accuracy, ClassBalancedFocalLoss
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
    parser.add_argument('--config', type=str, default='./config.json',
                        help='Path to the JSON configuration file')
    args = parser.parse_args()

    # Loading JSON configuration file
    print(f"Loading configuration file: {args.config}")
    with open(args.config, 'r') as f:
        config = json.load(f)

    # 1.1 Hyperparameters
    BATCH_SIZE = config['batch_size']
    NUM_EPOCHS = config['num_epochs']
    LR_BACKBONE = config.get('learning_rate_backbone', 1e-4)
    LR_HEAD = config.get('learning_rate_head', 5e-4)
    EARLY_STOPPING_PATIENCE = config.get('early_stopping_patience', 5)

    # 1.2 Dataset & Paths
    NUM_CLASSES = config['num_classes']
    DATA_DIR = config['data_dir']
    CHECKPOINT_PATH = config['checkpoint_path']
    BEST_MODEL_PATH = config['best_model_path']
    RESUME_TRAINING = config['resume_training']

    # 1.3 Directory Initialization
    checkpoint_dir = os.path.dirname(CHECKPOINT_PATH)
    best_model_dir = os.path.dirname(BEST_MODEL_PATH)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    if best_model_dir:
        os.makedirs(best_model_dir, exist_ok=True)

    # 1.4 Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ==============================================================================
    # 2. Data Preprocessing & Loaders
    # ==============================================================================
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(448, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
        ),
        transforms.RandomRotation(45),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2))
    ])

    val_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    train_dataset = ImageDataset(
        root_dir=DATA_DIR, split="train", transform=train_transform)
    val_dataset = ImageDataset(
        root_dir=DATA_DIR, split="val", transform=val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )

    # ==============================================================================
    # 3. Model, Loss, Optimizer & Scheduler
    # ==============================================================================
    # 3.1 Model Initialization
    model = ImageClassificationModel(
        num_classes=100, pretrained=True).to(device)

    for param in model.backbone_l1_l3[:5].parameters():
        param.requires_grad = False
    for param in model.backbone_l1_l3[5:].parameters():
        param.requires_grad = True
    for param in model.backbone_l4.parameters():
        param.requires_grad = True

    # 3.2 Loss Function : class-balance loss
    beta = 0.999
    train_labels = train_dataset.targets
    class_sample_count = np.bincount(train_labels, minlength=NUM_CLASSES)

    # Formula： (1 - beta) / (1 - beta^n)
    effective_num = 1.0 - np.power(beta, class_sample_count)
    effective_num = np.maximum(effective_num, 1e-8)  # avoid division by zero
    cb_weights = (1.0 - beta) / np.array(effective_num)
    cb_weights = cb_weights / np.sum(cb_weights) * NUM_CLASSES  #

    class_weights = torch.FloatTensor(cb_weights).to(device)
    criterion = ClassBalancedFocalLoss(
        cb_weights=class_weights, gamma=2.0, label_smoothing=0.1)

    # 3.3 Optimizer (Layer-wise LR)
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = optim.AdamW([
        # --- Backbone Params---
        {'params': model.backbone_l1_l3[5:].parameters(), 'lr': LR_BACKBONE},
        {'params': model.backbone_l4.parameters(), 'lr': LR_BACKBONE},

        # --- Head & Attention Params ---
        {'params': model.se_l3.parameters(), 'lr': LR_HEAD},
        {'params': model.gem.parameters(), 'lr': LR_HEAD},
        {'params': model.reduce3.parameters(), 'lr': LR_HEAD},
        {'params': model.reduce4.parameters(), 'lr': LR_HEAD},
        {'params': model.embedding.parameters(), 'lr': LR_HEAD},
        {'params': model.classifier.parameters(), 'lr': LR_HEAD}
    ], weight_decay=3e-4)

    # 3.4 Scheduler
    from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
    warmup_epochs = 5
    cosine_epochs = NUM_EPOCHS - warmup_epochs
    warmup_sch = LinearLR(
        optimizer,
        start_factor=0.5,
        total_iters=warmup_epochs
    )
    cosine_sch = CosineAnnealingLR(
        optimizer,
        T_max=cosine_epochs,
        eta_min=1e-6
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_sch, cosine_sch],
        milestones=[warmup_epochs]
    )

    if not model.check_parameters():
        return

    # 3.5 AMP Scaler
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

        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(" Loaded scheduler state.")

        epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
        best_val_preds = checkpoint.get('best_val_preds', [])
        best_val_labels = checkpoint.get('best_val_labels', [])
        print(
            f" Successfully loaded checkpoint! Resuming from Epoch {start_epoch+1}.")
    else:
        print("\n Starting training from scratch.")

    # ==============================================================================
    # 5. Main Training Loop
    # ==============================================================================
    training_start_time = time.time()
    try:
        for epoch in range(start_epoch, NUM_EPOCHS):
            print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")

            # 5.1 Train & Validate
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device, scaler)
            val_loss, val_acc, val_preds, val_labels = validate_one_epoch(
                model, val_loader, criterion, device)

            # 5.2 Step Scheduler
            scheduler.step()

            # 5.3 Record History
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            # 5.4 Print Metrics
            lr_backbone = optimizer.param_groups[0]['lr']
            lr_head = optimizer.param_groups[2]['lr']
            print(f"LR_Backbone: {lr_backbone:.6f} | LR_Head: {lr_head:.6f}")
            print(
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

            # 5.5 Early Stopping & Model Saving Logic
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

            # Save Checkpoint
            checkpoint_data = {
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
        # 6.1 Plot Curves
        plot_training_curves(
            history['train_loss'], history['val_loss'], history['train_acc'], history['val_acc'])

        # 6.2 Plot Diagnostics
        if best_val_preds and best_val_labels:
            plot_per_class_error(best_val_preds, best_val_labels,
                                 num_classes=NUM_CLASSES, save_path="./Plot/error_dist.png")
            plot_long_tail_accuracy(train_labels=train_dataset.targets, val_preds=best_val_preds,
                                    val_labels=best_val_labels, num_classes=NUM_CLASSES, save_path="./Plot/long_tail_acc.png")
            print(" Plots saved to ./Plot/")
    else:
        print("Training interrupted early. No plots generated.")

    print(
        f"\n Training completed. Best Val Acc: {best_val_acc:.2f}% | Time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")


if __name__ == "__main__":
    main()
