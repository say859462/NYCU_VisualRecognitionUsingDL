import torch
from tqdm import tqdm
from utils import make_background_suppressed_views


def train_one_epoch(
    model,
    train_loader,
    criterion,
    epoch,
    optimizer,
    device,
    scaler,
    stage,
    use_bg_suppression=False,
    bg_aux_weight=0.20,
    proto_div_weight=0.01,
    max_grad_norm=5.0,
):
    model.train()
    running_loss, correct_preds, total_preds = 0.0, 0, 0

    pbar = tqdm(
        train_loader,
        desc=f"Training Epoch {epoch}",
        leave=False,
        colour="blue"
    )

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        do_bg_aux = use_bg_suppression and stage == 1

        if do_bg_aux:
            with torch.no_grad():
                saliency = model.get_saliency(images)

            bg_views = make_background_suppressed_views(
                images,
                saliency,
                threshold_ratio=0.45,
                suppress_strength=0.35,
                blur_kernel=7
            ).to(device)
        else:
            bg_views = None

        use_amp = (device.type == 'cuda' and stage == 1)

        with torch.amp.autocast('cuda', enabled=use_amp):
            logits_full = model(images)
            loss = criterion(logits_full, labels)

            if do_bg_aux and bg_views is not None:
                logits_bg = model(bg_views)
                loss_bg = criterion(logits_bg, labels)
                loss = loss + bg_aux_weight * loss_bg

            loss_proto = model.prototype_diversity_loss(margin=0.2)
            loss = loss + proto_div_weight * loss_proto

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