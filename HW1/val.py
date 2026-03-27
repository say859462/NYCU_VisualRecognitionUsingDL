import torch
from tqdm import tqdm
from train import _get_stage_weights, _compute_pmg_loss


def _get_eval_logits(outputs, stage_cfg):
    if stage_cfg["fusion_weight"] > 0:
        return outputs["fusion_logits"]
    if stage_cfg["concat_weight"] > 0:
        return outputs["concat_logits"]
    return outputs["global_logits"]


def validate_one_epoch(model, val_loader, criterion, device, config, epoch):
    model.eval()

    stage_cfg = _get_stage_weights(
        epoch=epoch,
        stage1_epochs=config.get("pmg_stage1_epochs", 4),
        stage2_epochs=config.get("pmg_stage2_epochs", 4),
        config=config,
    )

    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []
    fusion_weights_snapshot = None

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation", leave=False)
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model.forward_pmg(images, stage_cfg=stage_cfg)
            loss = _compute_pmg_loss(outputs, labels, criterion, stage_cfg)

            logits_for_eval = _get_eval_logits(outputs, stage_cfg)
            preds = torch.argmax(logits_for_eval, dim=1)

            running_loss += loss.item() * images.size(0)
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            if fusion_weights_snapshot is None:
                fusion_weights_snapshot = outputs["fusion_weights"]

    return (
        running_loss / total,
        (correct / total) * 100,
        all_preds,
        all_labels,
        fusion_weights_snapshot,
    )
