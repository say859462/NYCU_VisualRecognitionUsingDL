import os
import random
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import torchvision.transforms.functional as TF
from torch.utils.data.sampler import Sampler
from collections import defaultdict
import copy
import pandas as pd


class PKSampler(Sampler):
    def __init__(self, labels, p, k):
        self.labels = labels
        self.p = p
        self.k = k
        self.batch_size = p * k
        self.label_to_indices = defaultdict(list)
        for i, label in enumerate(labels):
            self.label_to_indices[label].append(i)
        self.classes = list(self.label_to_indices.keys())

    def __iter__(self):
        label_to_indices = copy.deepcopy(self.label_to_indices)
        for label in self.classes:
            np.random.shuffle(label_to_indices[label])

        classes = copy.deepcopy(self.classes)
        np.random.shuffle(classes)

        num_batches = len(self.labels) // self.batch_size

        for _ in range(num_batches):
            if len(classes) < self.p:
                classes = copy.deepcopy(self.classes)
                np.random.shuffle(classes)

            sampled_classes = classes[:self.p]
            classes = classes[self.p:]

            batch = []
            for c in sampled_classes:
                if len(label_to_indices[c]) >= self.k:
                    indices = label_to_indices[c][:self.k]
                    label_to_indices[c] = label_to_indices[c][self.k:]
                else:
                    indices = np.random.choice(
                        self.label_to_indices[c], self.k, replace=True
                    ).tolist()
                batch.extend(indices)
            yield batch

    def __len__(self):
        return len(self.labels) // self.batch_size


class BalancedSoftmaxLoss(nn.Module):
    def __init__(self, sample_per_class):
        super().__init__()
        sample_per_class = torch.as_tensor(sample_per_class, dtype=torch.float32)
        self.register_buffer(
            "log_prior",
            torch.log(sample_per_class.clamp(min=1.0)).view(1, -1)
        )

    def forward(self, logits, targets):
        balanced_logits = logits + self.log_prior.to(logits.device)
        return F.cross_entropy(balanced_logits, targets)


def load_hard_classes_from_csv(csv_path, top_k_pairs=10):
    """
    從 top_confused_pairs.csv 取出最常混淆的類別集合
    需要欄位: True_Class, Pred_Class
    """
    if (csv_path is None) or (not os.path.exists(csv_path)):
        print(f"⚠️ hard-pair csv not found: {csv_path}, fallback to normal sampling")
        return []

    df = pd.read_csv(csv_path)
    required_cols = {"True_Class", "Pred_Class"}
    if not required_cols.issubset(set(df.columns)):
        print(f"⚠️ hard-pair csv missing columns {required_cols}, fallback to normal sampling")
        return []

    df = df.head(top_k_pairs)
    hard_classes = set(df["True_Class"].tolist()) | set(df["Pred_Class"].tolist())
    hard_classes = sorted(list(hard_classes))
    print(f"🎯 Hard classes from csv: {hard_classes}")
    return hard_classes


def build_pair_aware_sample_weights(labels, hard_classes, boost=1.8):
    """
    對 hard classes 給較高抽樣權重，其餘維持 1.0
    """
    weights = np.ones(len(labels), dtype=np.float32)
    if not hard_classes:
        return torch.tensor(weights, dtype=torch.float32)

    hard_set = set(hard_classes)
    for i, y in enumerate(labels):
        if int(y) in hard_set:
            weights[i] = boost
    return torch.tensor(weights, dtype=torch.float32)


def make_attention_crops(images, saliency_maps, out_size=None, threshold_ratio=0.6, pad_ratio=0.15):
    """
    根據 saliency 生成 crop，保留 full image + local crop 的雙視角訓練
    images: [B, C, H, W]
    saliency_maps: [B, Hs, Ws]
    """
    device = images.device
    b, c, h, w = images.shape

    if out_size is None:
        out_size = (h, w)

    crops = []
    saliency_maps = saliency_maps.detach()

    # 對齊到輸入影像大小
    if saliency_maps.dim() == 3:
        saliency_maps = saliency_maps.unsqueeze(1)  # [B,1,Hs,Ws]
    saliency_maps = F.interpolate(
        saliency_maps, size=(h, w), mode="bilinear", align_corners=False
    ).squeeze(1)  # [B,H,W]

    for i in range(b):
        sal = saliency_maps[i]
        sal = sal - sal.min()
        maxv = sal.max()

        if maxv.item() <= 1e-8:
            crop = images[i:i+1]
            crops.append(crop)
            continue

        sal = sal / maxv
        mask = sal >= (sal.max() * threshold_ratio)

        ys, xs = torch.where(mask)
        if ys.numel() == 0 or xs.numel() == 0:
            crop = images[i:i+1]
            crops.append(crop)
            continue

        y1, y2 = ys.min().item(), ys.max().item() + 1
        x1, x2 = xs.min().item(), xs.max().item() + 1

        box_h = y2 - y1
        box_w = x2 - x1

        pad_h = max(1, int(box_h * pad_ratio))
        pad_w = max(1, int(box_w * pad_ratio))

        y1 = max(0, y1 - pad_h)
        y2 = min(h, y2 + pad_h)
        x1 = max(0, x1 - pad_w)
        x2 = min(w, x2 + pad_w)

        crop = images[i:i+1, :, y1:y2, x1:x2]
        crop = F.interpolate(crop, size=out_size, mode="bilinear", align_corners=False)
        crops.append(crop)

    return torch.cat(crops, dim=0).to(device)


def plot_class_distribution(data_dir, title="Dataset Class Distribution", output_path="./Figures"):
    os.makedirs(output_path, exist_ok=True)
    counts = [len(os.listdir(os.path.join(data_dir, c)))
              for c in sorted(os.listdir(data_dir), key=int)]
    plt.figure(figsize=(18, 6))
    plt.bar(range(len(counts)), counts, color="skyblue", edgecolor="black")
    plt.axhline(y=np.mean(counts), color="red", linestyle="--")
    plt.xlabel("Class ID", fontsize=12)
    plt.ylabel("Number of Images", fontsize=12)
    plt.xticks(range(len(counts)), rotation=90, fontsize=8)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(
        output_path, f"{title.replace(' ', '_')}_distribution.png"))
    plt.close()
    return counts


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path="./Plot/training_curves.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend()
    ax1.set_title('Training and Validation Loss', fontsize=14)

    ax2.plot(epochs, train_accs, 'b-', label='Train Acc')
    ax2.plot(epochs, val_accs, 'r-', label='Val Acc')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.legend()
    ax2.set_title('Training and Validation Accuracy', fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_per_class_error(all_preds, all_labels, num_classes=100, save_path="./Plot/class_error_dist.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    error_rates = [np.sum(np.array(all_preds)[np.array(all_labels) == c] != c) / np.sum(np.array(
        all_labels) == c) * 100 if np.sum(np.array(all_labels) == c) > 0 else 0 for c in range(num_classes)]

    plt.figure(figsize=(20, 6))
    plt.bar(range(num_classes), error_rates, color='salmon', edgecolor='black')
    plt.axhline(y=np.mean(error_rates), color='red', linestyle='--',
                label=f'Mean Error ({np.mean(error_rates):.1f}%)')

    plt.xlabel('Class ID', fontsize=12)
    plt.ylabel('Error Rate (%)', fontsize=12)
    plt.xticks(range(num_classes), rotation=90, fontsize=8)
    plt.title('Error Rate per Class', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return error_rates


def plot_long_tail_accuracy(train_labels, val_preds, val_labels, num_classes=100, save_path="./Plot/long_tail_acc.png"):
    train_counts, val_accs = np.bincount(
        train_labels, minlength=num_classes), np.zeros(num_classes)
    for c in range(num_classes):
        val_accs[c] = np.sum(np.array(val_preds)[np.array(val_labels) == c] == c) / np.sum(
            np.array(val_labels) == c) * 100 if np.sum(np.array(val_labels) == c) > 0 else 0
    sorted_idx = np.argsort(train_counts)[::-1]

    fig, ax1 = plt.subplots(figsize=(18, 6))
    ax1.bar(range(num_classes), train_counts[sorted_idx],
            color='skyblue', alpha=0.6, label='Train Image Count')
    ax1.set_xlabel('Class ID (Sorted by Image Count)', fontsize=12)
    ax1.set_ylabel('Number of Training Images', color='skyblue', fontsize=12)
    ax1.set_xticks(range(num_classes))
    ax1.set_xticklabels(sorted_idx, rotation=90, fontsize=7)

    ax2 = ax1.twinx()
    ax2.plot(range(num_classes), val_accs[sorted_idx],
             color='red', marker='o', markersize=4, label='Val Accuracy')
    ax2.set_ylabel('Validation Accuracy (%)', color='red', fontsize=12)

    plt.title('Long Tail Accuracy Distribution', fontsize=14)
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_correlation_analysis(train_counts, error_rates, output_path="./Plot/correlation_analysis.png"):
    corr, p = pearsonr(train_counts, error_rates)
    plt.figure(figsize=(10, 7))
    plt.scatter(train_counts, error_rates, alpha=0.6, color='darkblue')
    z = np.polyfit(train_counts, error_rates, 1)
    plt.plot(train_counts, np.poly1d(z)(train_counts),
             "r--", label=f'Trend (r={corr:.2f})')
    plt.xlabel('Number of Training Images', fontsize=12)
    plt.ylabel('Error Rate (%)', fontsize=12)
    plt.title(f'Correlation Analysis (Pearson r: {corr:.4f})', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()