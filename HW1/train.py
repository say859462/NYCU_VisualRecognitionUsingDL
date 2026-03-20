import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn.utils as nn_utils

def train_one_epoch(model, train_loader, criterion_ce, criterion_ldam, cb_weights, epoch, optimizer, device, scaler, s=15.0, max_grad_norm=2.0):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    # ⭐ DRW 開關：前 15 輪建構基礎特徵，第 16 輪啟動顯微鏡與長尾雕刻
    enable_local = epoch > 15 

    pbar = tqdm(train_loader, desc=f"Training Epoch {epoch} (Local Path: {enable_local})", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            # 1. 第一階段 Forward：看全圖 (Global)
            logits_global, f4_global = model(images)

            if not enable_local:
                # Stage 1 (Epoch 1~15): 無壓力學習基礎特徵 (CE Loss)
                loss = criterion_ce(logits_global * s, labels)
            
            else:
                # Stage 2 (Epoch 16~30): 啟動動態裁切與 LDAM Loss
                
                # --- 動態裁切邏輯開始 ---
                with torch.no_grad():
                    B, C, H, W = images.shape
                    attn_map = f4_global.mean(dim=1)  # [B, H_f, W_f]

                    local_imgs = []
                    for i in range(B):
                        amap = attn_map[i]
                        # 正規化以便尋找高亮區
                        amap = (amap - amap.min()) / (amap.max() - amap.min() + 1e-6)
                        mask = amap > (amap.mean() + 0.5 * amap.std())

                        nonzero = torch.nonzero(mask)
                        if len(nonzero) == 0:  
                            local_imgs.append(F.interpolate(images[i:i+1], size=(448, 448), mode='bilinear')[0])
                            continue

                        y_min, x_min = nonzero.min(dim=0)[0]
                        y_max, x_max = nonzero.max(dim=0)[0]

                        # 映射回 512x512 的原圖座標
                        scale_y, scale_x = H / amap.shape[0], W / amap.shape[1]
                        y_min, y_max = int(y_min * scale_y), int(y_max * scale_y)
                        x_min, x_max = int(x_min * scale_x), int(x_max * scale_x)

                        # 向外 Padding 15%，保留蝴蝶觸角、鳥喙等邊緣特徵
                        h_box, w_box = y_max - y_min, x_max - x_min
                        y_min = max(0, int(y_min - 0.15 * h_box))
                        y_max = min(H, int(y_max + 0.15 * h_box))
                        x_min = max(0, int(x_min - 0.15 * w_box))
                        x_max = min(W, int(x_max + 0.15 * w_box))

                        # 從原圖上切下局部，並放大回 448x448
                        crop = images[i:i+1, :, y_min:y_max, x_min:x_max]
                        crop_resized = F.interpolate(crop, size=(448, 448), mode='bilinear', align_corners=False)[0]
                        local_imgs.append(crop_resized)

                    x_local = torch.stack(local_imgs)
                # --- 動態裁切邏輯結束 ---

                # 2. 第二次 Forward：看局部放大圖 (Local)
                logits_local, _ = model(x_local)

                # 3. 聯合 Loss (LDAM 雕刻長尾)
                # Global 和 Local 共用 classifier 的權重來推擠 Margin
                w_classifier = model.classifier.weight
                loss_global = criterion_ldam(logits_global, labels, w_classifier, cb_weights)
                loss_local = criterion_ldam(logits_local, labels, w_classifier, cb_weights)

                # 60% 關注整體輪廓，40% 關注顯微紋理
                loss = 0.6 * loss_global + 0.4 * loss_local

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn_utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(logits_global, 1)
        correct_preds += torch.sum(preds == labels.data).item()
        total_preds += labels.size(0)

        pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

    return running_loss / total_preds, (correct_preds / total_preds) * 100