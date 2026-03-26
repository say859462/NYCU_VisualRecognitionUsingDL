import torch
from tqdm import tqdm
from train import generate_cross_attention_bbox_local_view


def validate_one_epoch(model, val_loader, criterion, device, config, current_epoch):
    model.eval()

    running_loss = 0.0
    total_preds = 0
    correct_active = 0
    correct_full = 0
    correct_local = 0
    correct_fused = 0

    all_predictions = []
    all_targets = []

    stage_b_start_epoch = config.get("stage_b_start_epoch", 10)
    in_stage_b = current_epoch >= stage_b_start_epoch

    pbar = tqdm(val_loader, desc="Validating", leave=False, colour="green")

    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if not in_stage_b:
                full_logits = model(images)
                loss = criterion(full_logits, labels)

                full_preds = torch.argmax(full_logits, dim=1)

                batch_size = images.size(0)
                running_loss += loss.item() * batch_size
                total_preds += batch_size
                correct_active += torch.sum(full_preds == labels).item()
                correct_full += torch.sum(full_preds == labels).item()

                all_predictions.extend(full_preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

            else:
                local1_views = generate_cross_attention_bbox_local_view(
                    model=model,
                    images=images,
                    threshold_ratio=config['local_crop_threshold'],
                    padding_ratio=config['local_crop_padding_ratio'],
                    min_crop_ratio=config['local_min_crop_ratio'],
                    max_crop_ratio=config['local_max_crop_ratio'],
                    fallback_crop_ratio=config['local_fallback_crop_ratio']
                )

                outputs = model.forward_full_local(images, local1_views)

                full_logits = outputs["full_logits"]
                local_logits = outputs["local1_logits"]
                fused_logits = outputs["fused_logits"]

                loss = criterion(fused_logits, labels)

                full_preds = torch.argmax(full_logits, dim=1)
                local_preds = torch.argmax(local_logits, dim=1)
                fused_preds = torch.argmax(fused_logits, dim=1)

                batch_size = images.size(0)
                running_loss += loss.item() * batch_size
                total_preds += batch_size

                correct_active += torch.sum(fused_preds == labels).item()
                correct_full += torch.sum(full_preds == labels).item()
                correct_local += torch.sum(local_preds == labels).item()
                correct_fused += torch.sum(fused_preds == labels).item()

                all_predictions.extend(fused_preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

            pbar.set_postfix({
                'Loss': f"{running_loss / total_preds:.4f}",
                'Acc': f"{(correct_active / total_preds) * 100:.2f}%",
                'Stage': "B" if in_stage_b else "A"
            })

    epoch_loss = running_loss / total_preds
    epoch_acc = (correct_active / total_preds) * 100

    metrics = {
        "stage": "B" if in_stage_b else "A",
        "full_acc": (correct_full / total_preds) * 100,
        "local_acc": None if not in_stage_b else (correct_local / total_preds) * 100,
        "fused_acc": None if not in_stage_b else (correct_fused / total_preds) * 100,
    }

    return epoch_loss, epoch_acc, all_predictions, all_targets, metrics
