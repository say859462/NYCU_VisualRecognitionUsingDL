import torch
from tqdm import tqdm
import torch.nn.utils as nn_utils

def train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, max_grad_norm=2.0):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            # 1. 單純的前向傳播
            logits, embed = model(images)
            
            # 2. ⭐ 計算標準 Cross Entropy Loss
            loss = criterion(logits, labels)

        # 3. 反向傳播
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn_utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(logits, 1)
        correct_preds += torch.sum(preds == labels.data).item()
        total_preds += labels.size(0)

        pbar.set_postfix({
            'Loss': f"{running_loss / total_preds:.4f}",
            'Acc': f"{(correct_preds / total_preds) * 100:.2f}%"
        })

    epoch_loss = running_loss / total_preds
    epoch_acc = (correct_preds / total_preds) * 100

    return epoch_loss, epoch_acc