"""Training helpers for progressive multi-granularity supervision."""

from typing import Dict

import torch
from tqdm import tqdm


def _get_stage_weights(epoch: int, stage1_epochs: int, stage2_epochs: int, config: Dict) -> Dict:
    """Return loss weights for the current training stage."""
    if epoch <= stage1_epochs:
        return {
            "stage_name": "Stage A | Uniform warmup",
            "global_weight": config.get("pmg_stage1_global_weight", 0.8),
            "part2_weight": config.get("pmg_stage1_part2_weight", 0.8),
            "part4_weight": config.get("pmg_stage1_part4_weight", 0.8),
            "concat_weight": config.get("pmg_stage1_concat_weight", 0.8),
            "use_concat_as_main": False,  
        }
    if epoch <= stage1_epochs + stage2_epochs:
        return {
            "stage_name": "Stage B | Uniform joint learning",
            "global_weight": config.get("pmg_stage2_global_weight", 1.0),
            "part2_weight": config.get("pmg_stage2_part2_weight", 1.0),
            "part4_weight": config.get("pmg_stage2_part4_weight", 1.0),
            "concat_weight": config.get("pmg_stage2_concat_weight", 1.0),
            "use_concat_as_main": False,
        }
    return {
        "stage_name": "Stage C | Final-centric refinement",
        "global_weight": config.get("pmg_stage3_global_weight", 0.7),
        "part2_weight": config.get("pmg_stage3_part2_weight", 0.7),
        "part4_weight": config.get("pmg_stage3_part4_weight", 0.7),
        "concat_weight": config.get("pmg_stage3_concat_weight", 1.0),
        "use_concat_as_main": False,
    }


def _compute_pmg_loss(outputs, labels, criterion, stage_cfg):
    """Compute the weighted PMG classification loss for one batch."""
    loss = outputs["global_logits"].new_tensor(0.0)

    if stage_cfg["global_weight"] > 0:
        loss = loss + stage_cfg["global_weight"] * criterion(outputs["global_logits"], labels)
    if stage_cfg["part2_weight"] > 0:
        loss = loss + stage_cfg["part2_weight"] * criterion(outputs["part2_logits"], labels)
    if stage_cfg["part4_weight"] > 0:
        loss = loss + stage_cfg["part4_weight"] * criterion(outputs["part4_logits"], labels)
    if stage_cfg["concat_weight"] > 0:
        loss = loss + stage_cfg["concat_weight"] * criterion(outputs["concat_logits"], labels)

    return loss


def _get_eval_logits(outputs, stage_cfg):
    """Select which logits should be treated as the main prediction."""

    return outputs["global_logits"]


def _compute_batch_acc(logits, labels):
    """Compute correct predictions and predicted labels for a batch."""
    preds = torch.argmax(logits, dim=1)
    correct = torch.sum(preds == labels).item()
    return correct, preds


def train_one_epoch(model, train_loader, criterion, epoch, optimizer, device, scaler, config, max_grad_norm: float = 5.0):
    """Run one training epoch and return summary statistics."""
    model.train()
    stage_cfg = _get_stage_weights(
        epoch=epoch,
        stage1_epochs=config.get("pmg_stage1_epochs", 6),
        stage2_epochs=config.get("pmg_stage2_epochs", 10),
        config=config,
    )

    running_loss = 0.0
    total = 0
    main_correct = 0
    concat_correct = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        use_amp = device.type == "cuda"

        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model.forward_pmg(images)
            loss = _compute_pmg_loss(outputs, labels, criterion, stage_cfg)

        if not torch.isfinite(loss):
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.size(0)
        total += batch_size
        running_loss += loss.item() * batch_size

        # main_acc = global_acc
        logits_for_main = _get_eval_logits(outputs, stage_cfg)
        batch_main_correct, _ = _compute_batch_acc(logits_for_main, labels)
        main_correct += batch_main_correct

        # concat_acc = final_acc
        batch_concat_correct, _ = _compute_batch_acc(outputs["concat_logits"], labels)
        concat_correct += batch_concat_correct

        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "global_acc": f"{(main_correct / total) * 100:.2f}%",
                "concat_acc": f"{(concat_correct / total) * 100:.2f}%",
            }
        )

    return {
        "loss": running_loss / max(1, total),
        "main_acc": (main_correct / max(1, total)) * 100,      
        "concat_acc": (concat_correct / max(1, total)) * 100,  
        "stage_cfg": stage_cfg,
    }
