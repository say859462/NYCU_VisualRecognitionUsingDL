import torch
from tqdm import tqdm


def _get_stage_weights(epoch, stage1_epochs, stage2_epochs, config):
    if epoch <= stage1_epochs:
        return {
            "stage_name": "Stage 1 | Global anchor",
            "global_weight": config.get("pmg_stage1_global_weight", 1.0),
            "part2_weight": 0.0,
            "part4_weight": 0.0,
            "concat_weight": 0.0,
            "router_weight": 0.0,
        }

    if epoch <= stage1_epochs + stage2_epochs:
        return {
            "stage_name": "Stage 2 | Fusion-centric coarse learning",
            "global_weight": config.get("pmg_stage2_global_weight", 1.0),
            "part2_weight": config.get("pmg_stage2_part2_weight", 0.2),
            "part4_weight": config.get("pmg_stage2_part4_weight", 0.0),
            "concat_weight": config.get("pmg_stage2_concat_weight", 0.3),
            "router_weight": 0.0,
        }

    return {
        "stage_name": "Stage 3 | Fusion-centric fine learning",
        "global_weight": config.get("pmg_stage3_global_weight", 0.2),
        "part2_weight": config.get("pmg_stage3_part2_weight", 0.1),
        "part4_weight": config.get("pmg_stage3_part4_weight", 0.0),
        "concat_weight": config.get("pmg_stage3_concat_weight", 1.0),
        "router_weight": 0.0,
    }


def _compute_attention_regularization(outputs, config):
    if "fusion_attn" not in outputs:
        return 0.0

    fusion_attn = outputs["fusion_attn"]  # [B, T, T]
    cls_attn = fusion_attn[:, 0, :]       # [B, T]

    global_attn = cls_attn[:, 1]
    part2_attn_sum = cls_attn[:, 2:6].sum(dim=1)
    part4_attn_sum = cls_attn[:, 6:].sum(dim=1)
    local_attn_sum = part2_attn_sum + part4_attn_sum

    global_cap = config.get("fusion_global_attn_cap", 1.0)
    global_reg_w = config.get("fusion_global_attn_reg_weight", 0.0)

    local_floor = config.get("fusion_local_attn_floor", 0.0)
    local_reg_w = config.get("fusion_local_attn_reg_weight", 0.0)

    reg = 0.0

    if global_reg_w > 0:
        reg = reg + global_reg_w * torch.relu(global_attn - global_cap).pow(2).mean()

    if local_reg_w > 0:
        reg = reg + local_reg_w * torch.relu(local_floor - local_attn_sum).pow(2).mean()

    return reg


def _compute_pmg_loss(outputs, labels, criterion, stage_cfg, model=None, config=None):
    del model
    loss = 0.0

    if stage_cfg["global_weight"] > 0:
        loss = loss + stage_cfg["global_weight"] * criterion(outputs["global_logits"], labels)

    if stage_cfg["part2_weight"] > 0:
        loss = loss + stage_cfg["part2_weight"] * criterion(outputs["part2_logits"], labels)

    if stage_cfg["part4_weight"] > 0:
        loss = loss + stage_cfg["part4_weight"] * criterion(outputs["part4_logits"], labels)

    if stage_cfg["concat_weight"] > 0:
        loss = loss + stage_cfg["concat_weight"] * criterion(outputs["concat_logits"], labels)

    if config is not None and stage_cfg["concat_weight"] > 0:
        loss = loss + _compute_attention_regularization(outputs, config)

    return loss


def _get_eval_logits(outputs, stage_cfg):
    if stage_cfg["concat_weight"] > 0:
        return outputs["concat_logits"]
    return outputs["global_logits"]


def _compute_batch_acc(logits, labels):
    preds = torch.argmax(logits, dim=1)
    correct = torch.sum(preds == labels).item()
    return correct, preds


def train_one_epoch(model, train_loader, criterion, epoch, optimizer, device, scaler, config, max_grad_norm=5.0):
    model.train()

    stage_cfg = _get_stage_weights(
        epoch=epoch,
        stage1_epochs=config.get("pmg_stage1_epochs", 4),
        stage2_epochs=config.get("pmg_stage2_epochs", 4),
        config=config,
    )

    proto_diversity_weight = config.get("proto_diversity_weight", 0.0)

    running_loss = 0.0
    total = 0

    main_correct = 0
    concat_correct = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        use_amp = (device.type == "cuda")

        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model.forward_pmg(images)
            loss = _compute_pmg_loss(
                outputs, labels, criterion, stage_cfg, model=model, config=config
            )
            if proto_diversity_weight > 0:
                loss = loss + proto_diversity_weight * model.prototype_diversity_loss()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.size(0)
        total += batch_size
        running_loss += loss.item() * batch_size

        logits_for_main = _get_eval_logits(outputs, stage_cfg)
        batch_main_correct, _ = _compute_batch_acc(logits_for_main, labels)
        main_correct += batch_main_correct

        batch_concat_correct, _ = _compute_batch_acc(outputs["concat_logits"], labels)
        concat_correct += batch_concat_correct

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "main_acc": f"{(main_correct / total) * 100:.2f}%"
        })

    return {
        "loss": running_loss / total,
        "main_acc": (main_correct / total) * 100,
        "concat_acc": (concat_correct / total) * 100,
        "router_acc": 0.0,
        "stage_cfg": stage_cfg,
        "router_weight_means": None,
    }