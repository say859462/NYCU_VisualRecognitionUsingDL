import torch
from tqdm import tqdm

# ⭐ 加上 s=20.0 參數


def validate_one_epoch(model, val_loader, criterion, device, s=20.0):
    model.eval()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    all_predictions = []
    all_targets = []

    pbar = tqdm(val_loader, desc="Validating", leave=False, colour="green")
    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            outputs, _ = model(images)
            scaled_outputs = outputs * s

            loss = criterion(scaled_outputs, labels)

            running_loss += loss.item() * images.size(0)

            # ⭐ 改用 scaled_outputs 預測
            _, preds = torch.max(scaled_outputs, 1)

            correct_preds += torch.sum(preds == labels.data).item()
            total_preds += images.size(0)

            all_predictions.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

            pbar.set_postfix({
                'Loss': f"{running_loss / total_preds:.4f}",
                'Acc': f"{(correct_preds / total_preds) * 100:.2f}%"
            })

    epoch_loss = running_loss / total_preds
    epoch_acc = (correct_preds / total_preds) * 100

    return epoch_loss, epoch_acc, all_predictions, all_targets
