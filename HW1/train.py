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
            logits, embed = model(images)
            
            # ⭐ 重新加上 weight 以計算 LDAM 相似度懲罰
            weight = model.classifier.weight
            loss = criterion(logits, labels, weight)

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