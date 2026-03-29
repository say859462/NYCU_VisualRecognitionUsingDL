import torch
from tqdm import tqdm

from train import _compute_pmg_loss, _get_stage_weights


def _get_eval_logits(outputs, stage_cfg):
    if stage_cfg["concat_weight"] > 0:
        return outputs["concat_logits"]
    return outputs["global_logits"]


def _compute_batch_acc(logits, labels):
    preds = torch.argmax(logits, dim=1)
    correct = torch.sum(preds == labels).item()
    return correct, preds


def validate_one_epoch(model, val_loader, criterion, device, config, epoch):
    model.eval()

    stage_cfg = _get_stage_weights(
        epoch=epoch,
        stage1_epochs=config.get("pmg_stage1_epochs", 4),
        stage2_epochs=config.get("pmg_stage2_epochs", 4),
        config=config,
    )

    running_loss = 0.0
    total = 0

    main_correct = 0
    concat_correct = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation", leave=False)
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model.forward_pmg(images)
            loss = _compute_pmg_loss(
                outputs, labels, criterion, stage_cfg, model=model, config=config
            )

            batch_size = images.size(0)
            total += batch_size
            running_loss += loss.item() * batch_size

            logits_for_main = _get_eval_logits(outputs, stage_cfg)
            batch_main_correct, preds = _compute_batch_acc(logits_for_main, labels)
            main_correct += batch_main_correct

            batch_concat_correct, _ = _compute_batch_acc(outputs["concat_logits"], labels)
            concat_correct += batch_concat_correct

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    return {
        "loss": running_loss / total,
        "main_acc": (main_correct / total) * 100,
        "concat_acc": (concat_correct / total) * 100,
        "router_acc": 0.0,
        "preds": all_preds,
        "labels": all_labels,
        "router_weight_means": None,
    }