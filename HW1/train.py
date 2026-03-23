import torch
import numpy as np
from tqdm import tqdm
import torch.nn.utils as nn_utils


def train_one_epoch(model, train_loader, criterions, epoch, optimizer, device, scaler, max_grad_norm=5.0):
    model.train()
    running_loss, correct_preds, total_preds = 0.0, 0, 0
    criterion_ce = criterions['ce']

    do_saliency_cutmix = criterions.get('do_saliency_cutmix', False)
    w3, w4, w_fused = criterions.get('pmg_weights', (1.0, 1.0, 1.0))

    pbar = tqdm(
        train_loader, desc=f"Training Epoch {epoch}", leave=False, colour="blue")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        is_mixed = False
        # ⭐ 只有在 Epoch >= 15 且標記為開啟時才觸發，避免早期注意力不準時亂剪
        if do_saliency_cutmix and np.random.rand() < 0.5:
            is_mixed = True
            lam = np.random.beta(1.0, 1.0)
            saliency = model.get_saliency(images)
            B, H_f, W_f = saliency.shape
            H_img, W_img = images.shape[2], images.shape[3]
            rand_index = torch.randperm(B).to(device)
            target_a, target_b = labels, labels[rand_index]
            cut_rat = np.sqrt(1. - lam)
            cut_w = int(W_img * cut_rat)
            cut_h = int(H_img * cut_rat)

            # 精準裁切 Donor 圖片的高亮區塊
            for i in range(B):
                donor_idx = rand_index[i]
                # 尋找 Donor 注意力圖的最高點
                idx = torch.argmax(saliency[donor_idx].view(-1))
                cy_f = (idx // W_f).item()
                cx_f = (idx % W_f).item()

                # 將低解析度座標映射回 448x448 原圖
                cy = int((cy_f + 0.5) * (H_img / H_f))
                cx = int((cx_f + 0.5) * (W_img / W_f))

                bbx1 = np.clip(cx - cut_w // 2, 0, W_img)
                bby1 = np.clip(cy - cut_h // 2, 0, H_img)
                bbx2 = np.clip(cx + cut_w // 2, 0, W_img)
                bby2 = np.clip(cy + cut_h // 2, 0, H_img)

                # 將 Donor (rand_index) 的精華部位，貼到 Recipient (i) 上
                images[i, :, bby1:bby2, bbx1:bbx2] = images[donor_idx,
                                                            :, bby1:bby2, bbx1:bbx2]

                # 校正實際的 lambda (若裁切框超出邊界)
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W_img * H_img))

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out3, out4, out_fused, e_fused = model(images)

            if is_mixed:
                def mixed_criterion(output, target_a, target_b, lam):
                    return lam * criterion_ce(output * 30.0, target_a) + (1 - lam) * criterion_ce(output * 30.0, target_b)

                loss3 = mixed_criterion(out3, target_a, target_b, lam)
                loss4 = mixed_criterion(out4, target_a, target_b, lam)
                loss_fused = mixed_criterion(
                    out_fused, target_a, target_b, lam)

                loss = w3 * loss3 + w4 * loss4 + w_fused * loss_fused
            else:
                if isinstance(criterion_ce, torch.nn.CrossEntropyLoss):
                    loss3 = criterion_ce(
                        out3 * 30.0, labels) if w3 > 0 else 0.0
                    loss4 = criterion_ce(
                        out4 * 30.0, labels) if w4 > 0 else 0.0
                    loss_fused = criterion_ce(
                        out_fused * 30.0, labels) if w_fused > 0 else 0.0
                else:
                    # ⭐ 關鍵修正：CurricularFace 絕對不能在權重為 0 時呼叫，否則會汙染 EMA t 值
                    loss3 = criterion_ce(out3, labels) if w3 > 0 else 0.0
                    loss4 = criterion_ce(out4, labels) if w4 > 0 else 0.0
                    loss_fused = criterion_ce(
                        out_fused, labels) if w_fused > 0 else 0.0

                loss = w3 * loss3 + w4 * loss4 + w_fused * loss_fused

            if not is_mixed and criterions.get('use_supcon', False):
                loss_sup = criterions['supcon'](e_fused, labels)
                loss = loss + 0.5 * loss_sup

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)

        # 準確率計算 (若為 CutMix 批次，則預測 target_a 就算對，以追蹤基礎趨勢)
        _, preds = torch.max(out_fused, 1)  # ⭐ 確保使用 out_fused
        current_labels = target_a if is_mixed else labels  # ⭐ 建議改用 is_mixed 判斷
        correct_preds += torch.sum(preds == current_labels.data).item()
        total_preds += labels.size(0)

        pbar.set_postfix({'Loss': f"{loss.item():.4f}",
                         'Acc': f"{(correct_preds / total_preds) * 100:.2f}%"})

    return running_loss / total_preds, (correct_preds / total_preds) * 100
