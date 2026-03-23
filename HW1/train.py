import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

def train_one_epoch(model, train_loader, criterions, epoch, optimizer, device, scaler, max_grad_norm=5.0):
    model.train()
    running_loss, correct_preds, total_preds = 0.0, 0, 0
    criterion_ce = criterions['ce']

    do_attn_crop_drop = criterions.get('do_attn_crop_drop', False)
    w3, w4, w_fused = criterions.get('pmg_weights', (1.0, 1.0, 1.0))

    pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}", leave=False, colour="blue")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        # ⭐ 新增：Attention Crop & Drop (WS-DAN 核心機制)
        if do_attn_crop_drop and np.random.rand() < 0.5:
            with torch.no_grad():
                saliency = model.get_saliency(images)  # [B, H_f, W_f]
                # 將 Saliency 放大對齊原圖尺寸 (448x448)
                saliency_up = F.interpolate(
                    saliency.unsqueeze(1), size=(images.shape[2], images.shape[3]), 
                    mode='bilinear', align_corners=False
                ).squeeze(1)

                for i in range(images.size(0)):
                    sal_map = saliency_up[i]
                    threshold = sal_map.max() * 0.5  # 找出響應值超過一半的重點區域
                    mask = (sal_map > threshold).float()
                    nonzero = torch.nonzero(mask)
                    
                    if nonzero.numel() == 0:
                        continue
                    
                    y_min, x_min = torch.min(nonzero, dim=0)[0]
                    y_max, x_max = torch.max(nonzero, dim=0)[0]
                    
                    # 避免裁切框過小導致插值報錯
                    if (y_max - y_min) < 10 or (x_max - x_min) < 10:
                        continue

                    if np.random.rand() < 0.5:
                        # 🔍 Attention Crop: 放大關鍵特徵
                        cropped = images[i:i+1, :, y_min:y_max, x_min:x_max]
                        images[i:i+1] = F.interpolate(cropped, size=(images.shape[2], images.shape[3]), mode='bilinear', align_corners=False)
                    else:
                        # 🚫 Attention Drop: 抹除關鍵特徵，強迫尋找次要細節
                        images[i, :, y_min:y_max, x_min:x_max] = 0.0

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out3, out4, out_fused, e_fused = model(images)

            # 捨棄 CutMix 後，Loss 運算回歸單純
            if isinstance(criterion_ce, torch.nn.CrossEntropyLoss):
                loss3 = criterion_ce(out3 * 30.0, labels) if w3 > 0 else 0.0
                loss4 = criterion_ce(out4 * 30.0, labels) if w4 > 0 else 0.0
                loss_fused = criterion_ce(out_fused * 30.0, labels) if w_fused > 0 else 0.0
            else:
                # CurricularFace 絕對不能在權重為 0 時呼叫，否則會汙染 EMA t 值
                loss3 = criterion_ce(out3, labels) if w3 > 0 else 0.0
                loss4 = criterion_ce(out4, labels) if w4 > 0 else 0.0
                loss_fused = criterion_ce(out_fused, labels) if w_fused > 0 else 0.0

            loss = w3 * loss3 + w4 * loss4 + w_fused * loss_fused

            if criterions.get('use_supcon', False):
                loss_sup = criterions['supcon'](e_fused, labels)
                loss = loss + 0.5 * loss_sup

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