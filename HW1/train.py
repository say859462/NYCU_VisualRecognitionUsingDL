import torch
import torch.nn.functional as F
from tqdm import tqdm


def _build_box_from_peak(cy, cx, crop_h, crop_w, H, W):
    y1 = max(0, int(cy) - crop_h // 2)
    y2 = min(H, y1 + crop_h)
    y1 = max(0, y2 - crop_h)

    x1 = max(0, int(cx) - crop_w // 2)
    x2 = min(W, x1 + crop_w)
    x1 = max(0, x2 - crop_w)

    return y1, y2, x1, x2


def _expand_with_padding(y1, y2, x1, x2, H, W, padding_ratio):
    box_h = max(1, y2 - y1)
    box_w = max(1, x2 - x1)

    pad_h = max(1, int(box_h * padding_ratio))
    pad_w = max(1, int(box_w * padding_ratio))

    y1 = max(0, y1 - pad_h)
    y2 = min(H, y2 + pad_h)
    x1 = max(0, x1 - pad_w)
    x2 = min(W, x2 + pad_w)

    return y1, y2, x1, x2


def _clamp_box_size(
    y1, y2, x1, x2,
    H, W,
    min_crop_h, min_crop_w,
    max_crop_h, max_crop_w
):
    box_h = y2 - y1
    box_w = x2 - x1

    if box_h < min_crop_h:
        cy = (y1 + y2) // 2
        y1 = max(0, cy - min_crop_h // 2)
        y2 = min(H, y1 + min_crop_h)
        y1 = max(0, y2 - min_crop_h)

    if box_w < min_crop_w:
        cx = (x1 + x2) // 2
        x1 = max(0, cx - min_crop_w // 2)
        x2 = min(W, x1 + min_crop_w)
        x1 = max(0, x2 - min_crop_w)

    box_h = y2 - y1
    box_w = x2 - x1

    if box_h > max_crop_h:
        cy = (y1 + y2) // 2
        y1 = max(0, cy - max_crop_h // 2)
        y2 = min(H, y1 + max_crop_h)
        y1 = max(0, y2 - max_crop_h)

    if box_w > max_crop_w:
        cx = (x1 + x2) // 2
        x1 = max(0, cx - max_crop_w // 2)
        x2 = min(W, x1 + max_crop_w)
        x1 = max(0, x2 - max_crop_w)

    return y1, y2, x1, x2


def generate_cross_attention_bbox_local_view(
    model,
    images,
    threshold_ratio=0.55,
    padding_ratio=0.12,
    min_crop_ratio=0.20,
    max_crop_ratio=0.75,
    fallback_crop_ratio=0.40,
):
    with torch.no_grad():
        attn_map = model.get_cross_attention_map(images)   # [B,1,4,4]
        attn_map = F.interpolate(
            attn_map,
            size=images.shape[-2:],
            mode='bilinear',
            align_corners=False
        )

    B, _, H, W = attn_map.shape
    device = images.device

    attn_min = attn_map.amin(dim=(2, 3), keepdim=True)
    attn_max = attn_map.amax(dim=(2, 3), keepdim=True)
    norm_map = (attn_map - attn_min) / (attn_max - attn_min + 1e-8)

    min_crop_h = max(2, int(H * min_crop_ratio))
    min_crop_w = max(2, int(W * min_crop_ratio))
    max_crop_h = max(min_crop_h, int(H * max_crop_ratio))
    max_crop_w = max(min_crop_w, int(W * max_crop_ratio))
    fallback_crop_h = max(2, int(H * fallback_crop_ratio))
    fallback_crop_w = max(2, int(W * fallback_crop_ratio))

    local_list = []

    for b in range(B):
        single_map = norm_map[b, 0]
        peak_val = single_map.max()
        mask = single_map >= (peak_val * threshold_ratio)

        coords = torch.nonzero(mask, as_tuple=False)

        if coords.numel() == 0:
            flat_idx = torch.argmax(single_map.view(-1)).item()
            cy = flat_idx // W
            cx = flat_idx % W
            y1, y2, x1, x2 = _build_box_from_peak(
                cy, cx, fallback_crop_h, fallback_crop_w, H, W
            )
        else:
            ys = coords[:, 0]
            xs = coords[:, 1]

            y1 = int(ys.min().item())
            y2 = int(ys.max().item()) + 1
            x1 = int(xs.min().item())
            x2 = int(xs.max().item()) + 1

            y1, y2, x1, x2 = _expand_with_padding(
                y1, y2, x1, x2, H, W, padding_ratio
            )

            y1, y2, x1, x2 = _clamp_box_size(
                y1, y2, x1, x2,
                H, W,
                min_crop_h, min_crop_w,
                max_crop_h, max_crop_w
            )

            if (y2 - y1) < 2 or (x2 - x1) < 2:
                flat_idx = torch.argmax(single_map.view(-1)).item()
                cy = flat_idx // W
                cx = flat_idx % W
                y1, y2, x1, x2 = _build_box_from_peak(
                    cy, cx, fallback_crop_h, fallback_crop_w, H, W
                )

        crop = images[b:b + 1, :, y1:y2, x1:x2]
        crop = F.interpolate(
            crop,
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )
        local_list.append(crop)

    local = torch.cat(local_list, dim=0).to(device)
    return local


def train_one_epoch(
    model,
    train_loader,
    criterion,
    epoch,
    optimizer,
    device,
    scaler,

    stage_b_start_epoch=10,

    stage_a_full_loss_weight=1.0,
    stage_a_proto_diversity_weight=0.0,

    stage_b_fused_loss_weight=1.0,
    stage_b_full_loss_weight=0.5,
    stage_b_local_loss_weight=0.05,
    stage_b_proto_diversity_weight=0.0,

    local_crop_threshold=0.55,
    local_crop_padding_ratio=0.12,
    local_min_crop_ratio=0.20,
    local_max_crop_ratio=0.75,
    local_fallback_crop_ratio=0.40,

    max_grad_norm=5.0,
):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    in_stage_b = epoch >= stage_b_start_epoch
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        use_amp = (device.type == 'cuda')

        with torch.amp.autocast('cuda', enabled=use_amp):
            if not in_stage_b:
                # Stage A: only full/global backbone
                full_logits = model(images)

                loss = stage_a_full_loss_weight * criterion(full_logits, labels)

                if stage_a_proto_diversity_weight > 0:
                    loss = loss + stage_a_proto_diversity_weight * model.prototype_diversity_loss()

                preds = torch.argmax(full_logits, dim=1)

            else:
                # Stage B: late local refinement
                local1_images = generate_cross_attention_bbox_local_view(
                    model=model,
                    images=images,
                    threshold_ratio=local_crop_threshold,
                    padding_ratio=local_crop_padding_ratio,
                    min_crop_ratio=local_min_crop_ratio,
                    max_crop_ratio=local_max_crop_ratio,
                    fallback_crop_ratio=local_fallback_crop_ratio,
                )

                outputs = model.forward_full_local(images, local1_images)
                fused_logits = outputs["fused_logits"]
                full_logits = outputs["full_logits"]
                local1_logits = outputs["local1_logits"]

                loss = stage_b_fused_loss_weight * criterion(fused_logits, labels)

                if stage_b_full_loss_weight > 0:
                    loss = loss + stage_b_full_loss_weight * criterion(full_logits, labels)

                if stage_b_local_loss_weight > 0:
                    loss = loss + stage_b_local_loss_weight * criterion(local1_logits, labels)

                if stage_b_proto_diversity_weight > 0:
                    loss = loss + stage_b_proto_diversity_weight * model.prototype_diversity_loss()

                preds = torch.argmax(fused_logits, dim=1)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        correct += torch.sum(preds == labels).item()
        total += labels.size(0)

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{(correct / total) * 100:.2f}%",
            "stage": "B" if in_stage_b else "A"
        })

    return running_loss / total, (correct / total) * 100