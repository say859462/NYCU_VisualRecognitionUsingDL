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
        }

    if epoch <= stage1_epochs + stage2_epochs:
        return {
            "stage_name": "Stage 2 | Global + coarse parts",
            "global_weight": config.get("pmg_stage2_global_weight", 1.0),
            "part2_weight": config.get("pmg_stage2_part2_weight", 0.5),
            "part4_weight": config.get("pmg_stage2_part4_weight", 0.0),
            "concat_weight": config.get("pmg_stage2_concat_weight", 0.5),
        }

    return {
        "stage_name": "Stage 3 | Cross-fused backbone + L3-part4 residual correction",
        "global_weight": config.get("pmg_stage3_global_weight", 1.0),
        "part2_weight": config.get("pmg_stage3_part2_weight", 0.5),
        "part4_weight": config.get("pmg_stage3_part4_weight", 0.5),
        "concat_weight": config.get("pmg_stage3_concat_weight", 1.0),
    }


def _compute_pmg_loss(outputs, labels, criterion, stage_cfg):
    loss = 0.0
    if stage_cfg["global_weight"] > 0:
        loss = loss + stage_cfg["global_weight"] * \
            criterion(outputs["global_logits"], labels)
    if stage_cfg["part2_weight"] > 0:
        loss = loss + stage_cfg["part2_weight"] * \
            criterion(outputs["part2_logits"], labels)
    if stage_cfg["part4_weight"] > 0:
        loss = loss + stage_cfg["part4_weight"] * \
            criterion(outputs["part4_logits"], labels)
    if stage_cfg["concat_weight"] > 0:
        loss = loss + stage_cfg["concat_weight"] * \
            criterion(outputs["concat_logits"], labels)
    return loss


def _get_eval_logits(outputs, stage_cfg):
    if stage_cfg["concat_weight"] > 0:
        return outputs["concat_logits"]
    return outputs["global_logits"]


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
    correct = 0
    total = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        use_amp = (device.type == "cuda")

        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model.forward_pmg(images)
            loss = _compute_pmg_loss(outputs, labels, criterion, stage_cfg)
            if proto_diversity_weight > 0:
                loss = loss + proto_diversity_weight * model.prototype_diversity_loss()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        logits_for_acc = _get_eval_logits(outputs, stage_cfg)
        preds = torch.argmax(logits_for_acc, dim=1)

        running_loss += loss.item() * images.size(0)
        correct += torch.sum(preds == labels).item()
        total += labels.size(0)

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{(correct / total) * 100:.2f}%"
        })

    return running_loss / total, (correct / total) * 100, stage_cfg
