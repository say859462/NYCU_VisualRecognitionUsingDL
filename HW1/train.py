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
            "stage_name": "Stage 2 | Global + coarse parts",
            "global_weight": config.get("pmg_stage2_global_weight", 1.0),
            "part2_weight": config.get("pmg_stage2_part2_weight", 0.5),
            "part4_weight": config.get("pmg_stage2_part4_weight", 0.0),
            "concat_weight": config.get("pmg_stage2_concat_weight", 0.5),
            "router_weight": 0.0,
        }

    return {
        "stage_name": "Stage 3 | Pure PMG + sample-conditioned logit router",
        "global_weight": config.get("pmg_stage3_global_weight", 1.0),
        "part2_weight": config.get("pmg_stage3_part2_weight", 0.5),
        "part4_weight": config.get("pmg_stage3_part4_weight", 0.5),
        "concat_weight": config.get("pmg_stage3_concat_weight", 1.0),
        "router_weight": config.get("router_loss_weight", 1.0)
        if config.get("use_logit_router", True) else 0.0,
    }


def _compute_pmg_loss(outputs, labels, criterion, stage_cfg, model=None, config=None):
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

    if stage_cfg.get("router_weight", 0.0) > 0 and "router_logits" in outputs:
        loss = loss + stage_cfg["router_weight"] * \
            criterion(outputs["router_logits"], labels)

        if model is not None and config is not None:
            balance_w = config.get("router_balance_weight", 0.0)
            if balance_w > 0 and "router_weights" in outputs:
                loss = loss + balance_w * \
                    model.router_balance_loss(outputs["router_weights"])

    return loss


def _get_eval_logits(outputs, stage_cfg):
    if stage_cfg.get("router_weight", 0.0) > 0 and "router_logits" in outputs:
        return outputs["router_logits"]
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
    router_correct = 0

    router_weight_sum = None
    router_weight_count = 0

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

        # main acc
        logits_for_main = _get_eval_logits(outputs, stage_cfg)
        batch_main_correct, _ = _compute_batch_acc(logits_for_main, labels)
        main_correct += batch_main_correct

        # concat acc
        batch_concat_correct, _ = _compute_batch_acc(
            outputs["concat_logits"], labels)
        concat_correct += batch_concat_correct

        # router acc
        if "router_logits" in outputs:
            batch_router_correct, _ = _compute_batch_acc(
                outputs["router_logits"], labels)
            router_correct += batch_router_correct

        # router weights mean
        if "router_weights" in outputs:
            batch_router_mean = outputs["router_weights"].detach().mean(dim=0)
            if router_weight_sum is None:
                router_weight_sum = torch.zeros_like(batch_router_mean)
            router_weight_sum += batch_router_mean
            router_weight_count += 1

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "main_acc": f"{(main_correct / total) * 100:.2f}%"
        })

    mean_router_weights = None
    if router_weight_sum is not None and router_weight_count > 0:
        mean_router_weights = (router_weight_sum /
                               router_weight_count).detach().cpu()
        mean_router_weights = {
            "global": float(mean_router_weights[0].item()),
            "part2": float(mean_router_weights[1].item()),
            "part4": float(mean_router_weights[2].item()),
            "concat": float(mean_router_weights[3].item()),
        }

    return {
        "loss": running_loss / total,
        "main_acc": (main_correct / total) * 100,
        "concat_acc": (concat_correct / total) * 100,
        "router_acc": (router_correct / total) * 100 if "use_logit_router" in config and config.get("use_logit_router", True) else 0.0,
        "stage_cfg": stage_cfg,
        "router_weight_means": mean_router_weights,
    }
