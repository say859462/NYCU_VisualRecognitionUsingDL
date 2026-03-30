"""Utility functions for visualization, cropping, and analysis."""

import math
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.stats import pearsonr
from torchvision.transforms import functional as TF


class PadToSquare:
    """Pad a PIL image to a square canvas before resizing."""
    def __init__(self, fill=(0, 0, 0)):
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == h:
            return img

        max_side = max(w, h)
        pad_left = (max_side - w) // 2
        pad_right = max_side - w - pad_left
        pad_top = (max_side - h) // 2
        pad_bottom = max_side - h - pad_top
        return TF.pad(img, [pad_left, pad_top, pad_right, pad_bottom], fill=self.fill)


def _normalize_map(attn: torch.Tensor) -> torch.Tensor:
    """Normalize a saliency map to the [0, 1] range."""
    attn = attn - attn.amin(dim=(2, 3), keepdim=True)
    attn = attn / (attn.amax(dim=(2, 3), keepdim=True) + 1e-8)
    return attn


def build_attention_boxes(
    attention_map: torch.Tensor,
    threshold: float = 0.50,
    padding_ratio: float = 0.18,
    min_crop_ratio: float = 0.35,
    fallback_crop_ratio: float = 0.55,
) -> List[Tuple[int, int, int, int]]:
    if attention_map.dim() == 3:
        attention_map = attention_map.unsqueeze(1)

    attention_map = _normalize_map(attention_map)
    batch_size, _, height, width = attention_map.shape
    boxes = []

    for idx in range(batch_size):
        attn = attention_map[idx, 0]
        mask = attn >= threshold
        ys, xs = torch.where(mask)

        if ys.numel() == 0:
            crop_h = max(1, int(height * fallback_crop_ratio))
            crop_w = max(1, int(width * fallback_crop_ratio))
            top = max(0, (height - crop_h) // 2)
            left = max(0, (width - crop_w) // 2)
            boxes.append((top, left, top + crop_h, left + crop_w))
            continue

        y1 = int(ys.min().item())
        y2 = int(ys.max().item()) + 1
        x1 = int(xs.min().item())
        x2 = int(xs.max().item()) + 1

        box_h = y2 - y1
        box_w = x2 - x1
        pad_h = int(box_h * padding_ratio)
        pad_w = int(box_w * padding_ratio)

        y1 = max(0, y1 - pad_h)
        x1 = max(0, x1 - pad_w)
        y2 = min(height, y2 + pad_h)
        x2 = min(width, x2 + pad_w)

        min_h = max(1, int(height * min_crop_ratio))
        min_w = max(1, int(width * min_crop_ratio))

        cur_h = y2 - y1
        cur_w = x2 - x1

        if cur_h < min_h:
            extra = min_h - cur_h
            y1 = max(0, y1 - extra // 2)
            y2 = min(height, y2 + math.ceil(extra / 2))
        if cur_w < min_w:
            extra = min_w - cur_w
            x1 = max(0, x1 - extra // 2)
            x2 = min(width, x2 + math.ceil(extra / 2))

        if (y2 - y1) < min_h:
            y1 = max(0, min(height - min_h, y1))
            y2 = min(height, y1 + min_h)
        if (x2 - x1) < min_w:
            x1 = max(0, min(width - min_w, x1))
            x2 = min(width, x1 + min_w)

        boxes.append((y1, x1, y2, x2))

    return boxes


def crop_and_resize_batch(images: torch.Tensor, boxes: List[Tuple[int, int, int, int]]) -> torch.Tensor:
    """Crop each image with its box and resize it back to the input size."""
    crops = []
    _, _, height, width = images.shape
    for idx, (y1, x1, y2, x2) in enumerate(boxes):
        y1 = max(0, min(height - 1, y1))
        x1 = max(0, min(width - 1, x1))
        y2 = max(y1 + 1, min(height, y2))
        x2 = max(x1 + 1, min(width, x2))
        crop = images[idx:idx + 1, :, y1:y2, x1:x2]
        crop = F.interpolate(
            crop,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        crops.append(crop)
    return torch.cat(crops, dim=0)


def make_background_suppressed_views(
    images: torch.Tensor,
    saliency_maps: torch.Tensor,
    threshold_ratio: float = 0.45,
    suppress_strength: float = 0.35,
    blur_kernel: int = 7,
) -> torch.Tensor:
    if saliency_maps.dim() == 3:
        saliency_maps = saliency_maps.unsqueeze(1)

    saliency_maps = F.interpolate(
        saliency_maps,
        size=images.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    saliency_maps = _normalize_map(saliency_maps)

    fg_mask = (saliency_maps >= threshold_ratio).float()
    bg_mask = 1.0 - fg_mask
    pad = blur_kernel // 2
    blurred = F.avg_pool2d(images, kernel_size=blur_kernel, stride=1, padding=pad)
    suppressed_bg = (1.0 - suppress_strength) * images + suppress_strength * blurred
    return images * fg_mask + suppressed_bg * bg_mask


def plot_training_curves(
    train_losses,
    val_losses,
    train_accs,
    val_accs,
    save_path="./Plot/training_curves.png",
):
    """Plot train/validation loss and accuracy curves."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = list(range(1, len(train_losses) + 1))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(epochs, train_losses, "b-", label="Train Loss")
    ax1.plot(epochs, val_losses, "r-", label="Val Loss")
    min_val_loss = min(val_losses)
    min_loss_epoch = val_losses.index(min_val_loss) + 1
    ax1.scatter(min_loss_epoch, min_val_loss, color="gold", edgecolor="red", s=200, marker="*", zorder=5)
    ax1.annotate(
        f"Best Loss: {min_val_loss:.4f}\n(Epoch {min_loss_epoch})",
        xy=(min_loss_epoch, min_val_loss),
        xytext=(0, 25),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", alpha=0.8),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
    )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.set_title("Training and Validation Loss")

    ax2.plot(epochs, train_accs, "b-", label="Train Acc")
    ax2.plot(epochs, val_accs, "r-", label="Val Acc")
    max_val_acc = max(val_accs)
    max_acc_epoch = val_accs.index(max_val_acc) + 1
    ax2.scatter(max_acc_epoch, max_val_acc, color="gold", edgecolor="red", s=200, marker="*", zorder=5)
    ax2.annotate(
        f"Best Acc: {max_val_acc:.2f}%\n(Epoch {max_acc_epoch})",
        xy=(max_acc_epoch, max_val_acc),
        xytext=(0, -35),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=10,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", alpha=0.8),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
    )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.set_title("Training and Validation Accuracy")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_per_class_error(all_preds, all_labels, num_classes=100, save_path="./Plot/class_error_dist.png"):
    """Plot class-wise validation error rates."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    error_rates = []
    labels_np = np.array(all_labels)
    preds_np = np.array(all_preds)
    for class_id in range(num_classes):
        mask = labels_np == class_id
        if mask.sum() == 0:
            error_rates.append(0.0)
        else:
            error_rates.append(float((preds_np[mask] != class_id).mean() * 100.0))

    plt.figure(figsize=(20, 6))
    plt.bar(range(num_classes), error_rates, color="salmon", edgecolor="black")
    plt.axhline(y=np.mean(error_rates), color="red", linestyle="--", label=f"Mean Error ({np.mean(error_rates):.1f}%)")
    plt.xlabel("Class ID")
    plt.ylabel("Error Rate (%)")
    plt.title("Error Rate per Class")
    plt.xticks(range(num_classes), rotation=90, fontsize=8)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_long_tail_accuracy(
    train_dir,
    all_preds,
    all_labels,
    num_classes=100,
    save_path="./Plot/long_tail_accuracy.png",
):
    """Plot the relation between train-set size and validation accuracy."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    train_counts = []
    for class_id in range(num_classes):
        class_dir = os.path.join(train_dir, str(class_id))
        train_counts.append(len(os.listdir(class_dir)) if os.path.exists(class_dir) else 0)

    val_accs = []
    labels_np = np.array(all_labels)
    preds_np = np.array(all_preds)
    for class_id in range(num_classes):
        mask = labels_np == class_id
        if mask.sum() == 0:
            val_accs.append(0.0)
        else:
            val_accs.append(float((preds_np[mask] == class_id).mean() * 100.0))

    sorted_idx = np.argsort(np.array(train_counts))[::-1]
    sorted_counts = np.array(train_counts)[sorted_idx]
    sorted_accs = np.array(val_accs)[sorted_idx]

    plt.figure(figsize=(18, 7))
    ax1 = plt.gca()
    ax1.bar(range(num_classes), sorted_counts, color="lightblue", alpha=0.8)
    ax1.set_ylabel("Number of Training Images", color="skyblue")
    ax1.set_xlabel("Class ID (Sorted by Image Count)")
    ax1.tick_params(axis="y", labelcolor="skyblue")
    ax1.set_xticks(range(num_classes))
    ax1.set_xticklabels(sorted_idx, rotation=90, fontsize=8)

    ax2 = ax1.twinx()
    ax2.plot(range(num_classes), sorted_accs, "r-o", markersize=4)
    ax2.set_ylabel("Validation Accuracy (%)", color="red")
    ax2.tick_params(axis="y", labelcolor="red")
    plt.title("Long Tail Accuracy Distribution")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    corr = pearsonr(train_counts, val_accs)[0] if len(set(train_counts)) > 1 and len(set(val_accs)) > 1 else 0.0
    return corr
