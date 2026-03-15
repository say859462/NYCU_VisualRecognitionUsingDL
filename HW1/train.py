import torch
from tqdm import tqdm
import numpy as np


def train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, max_grad_norm=2.0):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    # tqdm progress bar
    pbar = tqdm(train_loader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        #  Mixup
        # lam = np.random.beta(alpha, alpha)
        # index = torch.randperm(images.size(0)).to(device)
        # mixed_images = lam * images + (1 - lam) * images[index]
        # labels_a, labels_b = labels, labels[index]

        # Original normal Forward Pass and Loss calculation
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        # Loss is averaged over the batch, so we multiply by batch size to get total loss for the batch
        # In case of the last batch which might be smaller, this will still work correctly
        running_loss += loss.item() * images.size(0)

        # Take the class with the highest probability as the predicted label
        _, preds = torch.max(outputs, 1)

        correct_preds += torch.sum(preds == labels.data).item()
        total_preds += labels.size(0)

        # tqdm progress bar postfix update
        pbar.set_postfix({
            'Loss': f"{running_loss / total_preds:.4f}",
            'Acc': f"{(correct_preds / total_preds) * 100:.2f}%"
        })

    epoch_loss = running_loss / total_preds
    epoch_acc = (correct_preds / total_preds) * 100  # Percentage

    return epoch_loss, epoch_acc
