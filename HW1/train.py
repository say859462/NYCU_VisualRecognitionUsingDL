import torch
from tqdm import tqdm


def train_one_epoch(model, train_loader, criterions, epoch, optimizer, device, scaler, max_grad_norm=5.0):
    model.train()
    running_loss, correct_preds, total_preds = 0.0, 0, 0
    criterion_ce = criterions['ce']
    w3, w4, w_fused = criterions.get('pmg_weights', (1.0, 1.0, 1.0))
    supcon_weight = criterions.get('supcon_weight', 0.0)

    pbar = tqdm(
        train_loader, desc=f"Training Epoch {epoch}", leave=False, colour="blue")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out3, out4, out_fused, e_fused = model(images)

            loss3 = criterion_ce(out3 * 30.0, labels) if w3 > 0 else 0.0
            loss4 = criterion_ce(out4 * 30.0, labels) if w4 > 0 else 0.0
            loss_fused = criterion_ce(
                out_fused * 30.0, labels) if w_fused > 0 else 0.0

            loss = w3 * loss3 + w4 * loss4 + w_fused * loss_fused

            if criterions.get('use_supcon', False) and supcon_weight > 0:
                loss_sup = criterions['supcon'](e_fused, labels)
                loss = loss + supcon_weight * loss_sup

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)

        # 準確率計算
        _, preds = torch.max(out_fused, 1)
        correct_preds += torch.sum(preds == labels.data).item()
        total_preds += labels.size(0)

        pbar.set_postfix({'Loss': f"{loss.item():.4f}",
                         'Acc': f"{(correct_preds / total_preds) * 100:.2f}%"})

    return running_loss / total_preds, (correct_preds / total_preds) * 100
