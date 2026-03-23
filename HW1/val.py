import torch
from tqdm import tqdm


def validate_one_epoch(model, val_loader, criterion, device):
    model.eval()
    running_loss, correct_preds, total_preds = 0.0, 0, 0
    all_predictions, all_targets = [], []

    pbar = tqdm(val_loader, desc="Validating", leave=False, colour="green")
    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            # ⭐ 直接取用 Logits
            outputs = model(images)
            scaled_outputs = outputs * 30.0
            loss = criterion(scaled_outputs, labels)

            running_loss += loss.item() * images.size(0)

            # 預測類別直接取 Cosine 最大值即可 (因為數值單調遞增)
            _, preds = torch.max(outputs, 1)

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
