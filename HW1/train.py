import torch
from tqdm import tqdm
import numpy as np
from utils import get_attention_crops


def train_one_epoch(model, train_loader, criterion, optimizer, device, scaler,
                    center_loss_cbp=None, center_loss_gem=None, max_grad_norm=2.0, use_crop=False):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            if use_crop:
                # --- 階段 1：全域圖 Pass ---
                outputs = model(images, return_attn=True)
                logits_cbp_full, logits_gem_full, embed_cbp_full, embed_gem_full, activation_maps = outputs

                # 取得分類器權重用於計算相似度
                weight_cbp = model.classifier_cbp.weight
                weight_gem = model.classifier_gem.weight

                # Soft Target LDAM Loss (OHEM 內建於其中)
                loss_cbp_full_ce = criterion(
                    logits_cbp_full, labels, weight_cbp)
                loss_gem_full_ce = criterion(
                    logits_gem_full, labels, weight_gem)

                # Center Loss 聚集同類特徵
                loss_cbp_center_full = center_loss_cbp(embed_cbp_full, labels)
                loss_gem_center_full = center_loss_gem(embed_gem_full, labels)

                loss_cbp_full = loss_cbp_full_ce + 0.1 * loss_cbp_center_full
                loss_gem_full = loss_gem_full_ce + 0.1 * loss_gem_center_full

                loss_full = (loss_cbp_full + 0.3 * loss_gem_full) / 1.3
                scaler.scale(loss_full).backward()

                # --- 階段 2：生成裁切圖 ---
                with torch.no_grad():
                    cropped_imgs = get_attention_crops(
                        images, activation_maps.detach(), threshold=0.5)

                # --- 階段 3：局部特徵圖 Pass ---
                logits_cbp_crop, logits_gem_crop, embed_cbp_crop, embed_gem_crop = model(
                    cropped_imgs)

                loss_cbp_crop_ce = criterion(
                    logits_cbp_crop, labels, weight_cbp)
                loss_gem_crop_ce = criterion(
                    logits_gem_crop, labels, weight_gem)
                loss_cbp_center_crop = center_loss_cbp(embed_cbp_crop, labels)
                loss_gem_center_crop = center_loss_gem(embed_gem_crop, labels)

                loss_cbp_crop = loss_cbp_crop_ce + 0.1 * loss_cbp_center_crop
                loss_gem_crop = loss_gem_crop_ce + 0.1 * loss_gem_center_crop

                loss_crop = (loss_cbp_crop + 0.3 * loss_gem_crop) / 1.3
                scaler.scale(loss_crop).backward()

                loss = loss_full + 0.5 * loss_crop
                outputs_pred = (logits_cbp_full * 0.8 + logits_gem_full * 0.2)
            else:
                # 未開啟裁切的普通訓練
                logits_cbp, logits_gem, embed_cbp, embed_gem = model(images)

                weight_cbp = model.classifier_cbp.weight
                weight_gem = model.classifier_gem.weight

                loss_cbp_ce = criterion(logits_cbp, labels, weight_cbp)
                loss_gem_ce = criterion(logits_gem, labels, weight_gem)

                loss_cbp_center = center_loss_cbp(embed_cbp, labels)
                loss_gem_center = center_loss_gem(embed_gem, labels)

                loss_cbp = loss_cbp_ce + 0.1 * loss_cbp_center
                loss_gem = loss_gem_ce + 0.1 * loss_gem_center

                loss = (loss_cbp + 0.3 * loss_gem) / 1.3
                scaler.scale(loss).backward()
                outputs_pred = (logits_cbp * 0.8 + logits_gem * 0.2)

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs_pred, 1)
        correct_preds += torch.sum(preds == labels.data).item()
        total_preds += labels.size(0)

        pbar.set_postfix({
            'Loss': f"{running_loss / total_preds:.4f}",
            'Acc': f"{(correct_preds / total_preds) * 100:.2f}%"
        })

    epoch_loss = running_loss / total_preds
    epoch_acc = (correct_preds / total_preds) * 100

    return epoch_loss, epoch_acc
