from typing import Dict, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils import build_attention_boxes, crop_and_resize_batch


def _get_stage_weights(epoch: int, stage1_epochs: int, stage2_epochs: int, config: Dict) -> Dict:
    if epoch <= stage1_epochs:
        return {
            "stage_name": "Stage 1 | Global anchor",
            "global_weight": config.get("pmg_stage1_global_weight", 1.0),
            "part2_weight": 0.0,
            "part4_weight": 0.0,
            "concat_weight": 0.0,
            "crop_cls_weight": 0.0,
            "consistency_weight": 0.0,
            "supcon_weight": 0.0,
            "ranking_weight": 0.0,
        }

    if epoch <= stage1_epochs + stage2_epochs:
        return {
            "stage_name": "Stage 2 | Global + crop-aware coarse fusion",
            "global_weight": config.get("pmg_stage2_global_weight", 1.0),
            "part2_weight": config.get("pmg_stage2_part2_weight", 0.3),
            "part4_weight": 0.0,
            "concat_weight": config.get("pmg_stage2_concat_weight", 0.6),
            "crop_cls_weight": config.get("crop_cls_weight_stage2", 0.15),
            "consistency_weight": config.get("consistency_weight_stage2", 0.10),
            "supcon_weight": 0.0,
            "ranking_weight": 0.0,
        }

    return {
        "stage_name": "Stage 3 | Pairwise fusion + crop consistency + optional metric regularization",
        "global_weight": config.get("pmg_stage3_global_weight", 0.2),
        "part2_weight": config.get("pmg_stage3_part2_weight", 0.1),
        "part4_weight": 0.0,
        "concat_weight": config.get("pmg_stage3_concat_weight", 1.0),
        "crop_cls_weight": config.get("crop_cls_weight_stage3", 0.15),
        "consistency_weight": config.get("consistency_weight_stage3", 0.10),
        "supcon_weight": config.get("supcon_weight_stage3", 0.02),
        "ranking_weight": config.get("ranking_weight_stage3", 0.02),
    }


def _compute_kl_consistency(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    student_log_prob = F.log_softmax(student_logits / temperature, dim=1)
    teacher_prob = F.softmax(teacher_logits / temperature, dim=1)
    loss = F.kl_div(student_log_prob, teacher_prob, reduction="batchmean")
    return loss * (temperature ** 2)


def _compute_pmg_loss(
    outputs: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    criterion,
    stage_cfg: Dict,
    crop_outputs: Optional[Dict[str, torch.Tensor]] = None,
    model=None,
    config: Optional[Dict] = None,
) -> torch.Tensor:
    del model
    loss = torch.zeros((), device=labels.device, dtype=torch.float32)

    if stage_cfg["global_weight"] > 0:
        loss = loss + stage_cfg["global_weight"] * criterion(outputs["global_logits"], labels)
    if stage_cfg["part2_weight"] > 0:
        loss = loss + stage_cfg["part2_weight"] * criterion(outputs["part2_logits"], labels)
    if stage_cfg["part4_weight"] > 0:
        loss = loss + stage_cfg["part4_weight"] * criterion(outputs["part4_logits"], labels)
    if stage_cfg["concat_weight"] > 0:
        loss = loss + stage_cfg["concat_weight"] * criterion(outputs["concat_logits"], labels)

    if crop_outputs is not None and stage_cfg.get("crop_cls_weight", 0.0) > 0:
        loss = loss + stage_cfg["crop_cls_weight"] * criterion(crop_outputs["concat_logits"], labels)

    if crop_outputs is not None and stage_cfg.get("consistency_weight", 0.0) > 0:
        temperature = float(config.get("consistency_temperature", 2.0))
        loss = loss + stage_cfg["consistency_weight"] * (
            0.5 * _compute_kl_consistency(outputs["concat_logits"], crop_outputs["concat_logits"].detach(), temperature)
            + 0.5 * _compute_kl_consistency(crop_outputs["concat_logits"], outputs["concat_logits"].detach(), temperature)
        )

    if crop_outputs is not None and config is not None and (
        stage_cfg.get("supcon_weight", 0.0) > 0 or stage_cfg.get("ranking_weight", 0.0) > 0
    ):
        loss = loss + outputs["concat_embed"].new_tensor(0.0)
        loss = loss + crop_outputs["concat_embed"].new_tensor(0.0)
        loss = loss + config.get("regularizer_global_scale", 1.0) * outputs["concat_embed"].new_tensor(0.0)

    return loss


def _get_eval_logits(outputs: Dict[str, torch.Tensor], stage_cfg: Dict) -> torch.Tensor:
    if stage_cfg["concat_weight"] > 0:
        return outputs["concat_logits"]
    return outputs["global_logits"]


def _compute_batch_acc(logits: torch.Tensor, labels: torch.Tensor):
    preds = torch.argmax(logits, dim=1)
    correct = torch.sum(preds == labels).item()
    return correct, preds


def _build_crop_view(outputs: Dict[str, torch.Tensor], images: torch.Tensor, config: Dict) -> torch.Tensor:
    boxes = build_attention_boxes(
        outputs["attention_map"].detach(),
        threshold=float(config.get("local_crop_threshold", 0.50)),
        padding_ratio=float(config.get("local_crop_padding_ratio", 0.18)),
        min_crop_ratio=float(config.get("local_min_crop_ratio", 0.35)),
        fallback_crop_ratio=float(config.get("local_fallback_crop_ratio", 0.55)),
    )
    return crop_and_resize_batch(images, boxes)


def train_one_epoch(
    model,
    train_loader,
    criterion,
    epoch,
    optimizer,
    device,
    scaler,
    config,
    max_grad_norm: float = 5.0,
):
    model.train()

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
    crop_concat_correct = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        use_amp = device.type == "cuda"

        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model.forward_pmg(images)
            crop_outputs = None

            if stage_cfg.get("crop_cls_weight", 0.0) > 0 or stage_cfg.get("consistency_weight", 0.0) > 0:
                crop_images = _build_crop_view(outputs, images, config)
                crop_outputs = model.forward_pmg(crop_images)

            loss = _compute_pmg_loss(
                outputs,
                labels,
                criterion,
                stage_cfg,
                crop_outputs=crop_outputs,
                model=model,
                config=config,
            )

            if crop_outputs is not None and (
                stage_cfg.get("supcon_weight", 0.0) > 0 or stage_cfg.get("ranking_weight", 0.0) > 0
            ):
                reg_loss = model.compute_stage3_regularization(
                    full_embed=outputs["concat_embed"],
                    crop_embed=crop_outputs["concat_embed"],
                    labels=labels,
                    crop_logits=crop_outputs["concat_logits"],
                    supcon_weight=stage_cfg.get("supcon_weight", 0.0),
                    ranking_weight=stage_cfg.get("ranking_weight", 0.0),
                    ranking_margin=float(config.get("ranking_margin", 0.15)),
                )
                loss = loss + reg_loss

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

        if crop_outputs is not None:
            batch_crop_correct, _ = _compute_batch_acc(crop_outputs["concat_logits"], labels)
            crop_concat_correct += batch_crop_correct

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "main_acc": f"{(main_correct / total) * 100:.2f}%",
        })

    return {
        "loss": running_loss / total,
        "main_acc": (main_correct / total) * 100,
        "concat_acc": (concat_correct / total) * 100,
        "crop_concat_acc": (crop_concat_correct / total) * 100 if total > 0 else 0.0,
        "stage_cfg": stage_cfg,
    }
