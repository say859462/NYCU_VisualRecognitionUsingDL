import torch
from tqdm import tqdm
from utils import make_random_resized_views, feature_distribution_kl


def train_one_epoch(
    model,
    train_loader,
    criterion,
    epoch,
    optimizer,
    device,
    scaler,
    stage,
    use_multiview_distill=False,
    alpha_mid=0.10,
    distill_temperature=1.0,
    max_grad_norm=5.0,
):
    model.train()
    running_loss, correct_preds, total_preds = 0.0, 0, 0

    pbar = tqdm(
        train_loader, desc=f"Training Epoch {epoch}", leave=False, colour="blue")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        do_distill = use_multiview_distill and stage == 1

        if do_distill:
            mid_views = make_random_resized_views(
                images,
                scale_range=(0.65, 0.85),
                out_size=(images.shape[-2], images.shape[-1])
            ).to(device)
        else:
            mid_views = None

        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            # full image branch
            full_feat, _ = model.forward_features(images)
            logits_full = model.classifier(full_feat)
            loss_cls = criterion(logits_full, labels)
            loss = loss_cls

            # distillation branch
            if do_distill and mid_views is not None:
                teacher_feat = full_feat.detach()  # 關鍵：teacher 不回傳 gradient
                mid_feat, _ = model.forward_features(mid_views)

                loss_mid = feature_distribution_kl(
                    mid_feat,
                    teacher_feat,
                    temperature=distill_temperature
                )

                loss = loss_cls + alpha_mid * loss_mid

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits_full, dim=1)
        correct_preds += torch.sum(preds == labels).item()
        total_preds += labels.size(0)

        pbar.set_postfix({
            'Loss': f"{loss.item():.4f}",
            'Acc': f"{(correct_preds / total_preds) * 100:.2f}%"
        })

    return running_loss / total_preds, (correct_preds / total_preds) * 100
