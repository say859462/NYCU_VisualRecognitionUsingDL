import torch
from tqdm import tqdm


def train_one_epoch(model, train_loader, criterion, epoch, optimizer, device, scaler, max_grad_norm=5.0):
    model.train()
    running_loss, correct_preds, total_preds = 0.0, 0, 0

    pbar = tqdm(
        train_loader, desc=f"Training Epoch {epoch}", leave=False, colour="blue")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1)
        correct_preds += torch.sum(preds == labels).item()
        total_preds += labels.size(0)

        pbar.set_postfix({
            'Loss': f"{loss.item():.4f}",
            'Acc': f"{(correct_preds / total_preds) * 100:.2f}%"
        })

    return running_loss / total_preds, (correct_preds / total_preds) * 100
