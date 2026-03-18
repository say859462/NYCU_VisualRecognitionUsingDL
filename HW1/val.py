import torch
from tqdm import tqdm

def validate_one_epoch(model, val_loader, criterion, device, s=20.0):
    """
    s: 縮放係數 (scale factor)，必須與 SimilarityLDAMLoss 中的 s 保持一致，
       確保 val_loss 與 train_loss 在同一個數量級。
    """
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

            # 取得 Ensemble Logits (來自 NormedLinear，數值在 [-1, 1])
            outputs = model(images)
            
            # ⭐ 修正：補上 s=20.0 的縮放，還原 Logits 尺度以計算正確的 Cross Entropy Loss
            scaled_outputs = outputs * s

            # 計算驗證 Loss
            loss = criterion(scaled_outputs, labels)

            running_loss += loss.item() * images.size(0)

            # 計算準確率
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
    epoch_acc = (correct_preds / total_preds) * 100  # Percentage

    return epoch_loss, epoch_acc, all_predictions, all_targets