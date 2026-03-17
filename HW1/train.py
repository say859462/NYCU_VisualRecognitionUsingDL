import torch
from tqdm import tqdm
import numpy as np
from utils import get_attention_crops

def train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, max_grad_norm=2.0, use_crop=False):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16): # 若硬體不支援請改 float16
            if use_crop:
                # --- 階段 1：全域圖 Pass ---
                logits_cbp_full, logits_gem_full, activation_maps = model(images, return_attn=True)
                loss_cbp_full = criterion(logits_cbp_full, labels)
                loss_gem_full = criterion(logits_gem_full, labels)
                # 組合 Loss：CBP 是主幹，GeM 負責輔助定位 (權重 0.5)
                loss_full = (loss_cbp_full + 0.5 * loss_gem_full) / 2.0
                
                scaler.scale(loss_full).backward() # 立即釋放計算圖

                # --- 階段 2：生成裁切圖 ---
                with torch.no_grad():
                    cropped_imgs = get_attention_crops(images, activation_maps.detach(), threshold=0.6)
                
                # --- 階段 3：局部特徵圖 Pass ---
                logits_cbp_crop, logits_gem_crop = model(cropped_imgs)
                loss_cbp_crop = criterion(logits_cbp_crop, labels)
                loss_gem_crop = criterion(logits_gem_crop, labels)
                loss_crop = (loss_cbp_crop + 0.5 * loss_gem_crop) / 2.0
                
                scaler.scale(loss_crop).backward()

                loss = loss_full + loss_crop
                outputs = (logits_cbp_full + logits_gem_full) / 2.0 # 用全域圖算 Accuracy
            else:
                # 未開啟裁切的普通訓練
                logits_cbp, logits_gem = model(images)
                loss_cbp = criterion(logits_cbp, labels)
                loss_gem = criterion(logits_gem, labels)
                loss = loss_cbp + 0.5 * loss_gem

                scaler.scale(loss).backward()
                outputs = (logits_cbp + logits_gem) / 2.0

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct_preds += torch.sum(preds == labels.data).item()
        total_preds += labels.size(0)

        pbar.set_postfix({
            'Loss': f"{running_loss / total_preds:.4f}",
            'Acc': f"{(correct_preds / total_preds) * 100:.2f}%"
        })

    epoch_loss = running_loss / total_preds
    epoch_acc = (correct_preds / total_preds) * 100

    return epoch_loss, epoch_acc