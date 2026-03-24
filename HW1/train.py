import torch
from tqdm import tqdm
from utils import make_attention_crops


def train_one_epoch(
    model,
    train_loader,
    criterion,
    epoch,
    optimizer,
    device,
    scaler,
    stage,
    dual_view=False,
    full_weight=0.7,
    crop_weight=0.3,
    max_grad_norm=5.0
):
    model.train()
    running_loss, correct_preds, total_preds = 0.0, 0, 0

    pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}", leave=False, colour="blue")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        if dual_view and stage == 1:
            saliency = model.get_saliency(images)
            crop_images = make_attention_crops(
                images,
                saliency,
                out_size=(images.shape[-2], images.shape[-1]),
                threshold_ratio=0.6,
                pad_ratio=0.15
            ).to(device)
        else:
            crop_images = None

        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            logits_full = model(images)
            loss_full = criterion(logits_full, labels)

            if dual_view and stage == 1 and crop_images is not None:
                logits_crop = model(crop_images)
                loss_crop = criterion(logits_crop, labels)
                loss = full_weight * loss_full + crop_weight * loss_crop
            else:
                loss = loss_full

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits_full, dim=1)
        correct_preds += torch.sum(preds == labels).item()
        total_preds += labels.size(0)

        pbar.set_postfix({
            'Loss': f"{loss.item():.4f}",
            'Acc': f"{(correct_preds / total_preds) * 100:.2f}%"
        })

    return running_loss / total_preds, (correct_preds / total_preds) * 100