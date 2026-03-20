import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import torchvision.transforms.functional as TF


class ResidualSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(
            2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * (1 + attn)


def get_cb_weights(labels_list, num_classes=100, beta=0.999):
    class_counts = np.bincount(labels_list, minlength=num_classes)
    cb_weights = [(1.0 - beta) / (1.0 - np.power(beta, count))
                  if count > 0 else 0.0 for count in class_counts]
    cb_weights = np.array(cb_weights)
    return torch.FloatTensor(cb_weights / np.sum(cb_weights) * num_classes)


class ClassBalancedFocalLoss(nn.Module):
    def __init__(self, cb_weights, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.cb_weights = cb_weights
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss_unweighted = F.cross_entropy(
            inputs, targets, label_smoothing=self.label_smoothing, reduction='none')
        pt = torch.exp(-ce_loss_unweighted)
        focal_weight = (1 - pt) ** self.gamma
        return (self.cb_weights[targets] * focal_weight * ce_loss_unweighted).mean()


class RandomDiscreteRotation:
    def __init__(self, angles=[0, 90, 270], weights=[0.7, 0.15, 0.15]):
        self.angles, self.weights = angles, weights

    def __call__(self, img):
        return TF.rotate(img, random.choices(self.angles, weights=self.weights)[0])

# 繪圖函數保持不變


def plot_class_distribution(data_dir, title="Dataset Class Distribution", output_path="./Figures"):
    os.makedirs(output_path, exist_ok=True)
    counts = [len(os.listdir(os.path.join(data_dir, c)))
              for c in sorted(os.listdir(data_dir), key=int)]
    plt.figure(figsize=(18, 6))
    plt.bar(range(len(counts)), counts, color="skyblue", edgecolor="black")
    plt.axhline(y=np.mean(counts), color="red", linestyle="--")
    plt.savefig(os.path.join(
        output_path, f"{title.replace(' ', '_')}_distribution.png"))
    plt.close()
    return counts


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path="./Plot/training_curves.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss')
    if val_losses:
        ax1.axvline(val_losses.index(min(val_losses)) +
                    1, color='red', linestyle=':')
    ax1.legend()
    ax1.set_title('Loss')
    ax2.plot(epochs, train_accs, 'b-', label='Train Acc')
    ax2.plot(epochs, val_accs, 'r-', label='Val Acc')
    if val_accs:
        ax2.axvline(val_accs.index(max(val_accs)) +
                    1, color='red', linestyle=':')
    ax2.legend()
    ax2.set_title('Accuracy')
    plt.savefig(save_path)
    plt.close()


def plot_per_class_error(all_preds, all_labels, num_classes=100, save_path="./Plot/class_error_dist.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    error_rates = [np.sum(np.array(all_preds)[np.array(all_labels) == c] != c) / np.sum(np.array(
        all_labels) == c) * 100 if np.sum(np.array(all_labels) == c) > 0 else 0 for c in range(num_classes)]
    plt.figure(figsize=(18, 6))
    plt.bar(range(num_classes), error_rates, color='salmon', edgecolor='black')
    plt.axhline(y=np.mean(error_rates), color='red', linestyle='--')
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
    fig, ax1 = plt.subplots(figsize=(16, 6))
    ax1.bar(range(num_classes),
            train_counts[sorted_idx], color='skyblue', alpha=0.6)
    ax2 = ax1.twinx()
    ax2.plot(range(num_classes),
             val_accs[sorted_idx], color='red', marker='o', markersize=4)
    plt.savefig(save_path)
    plt.close()


def plot_correlation_analysis(train_counts, error_rates, output_path="./Plot/correlation_analysis.png"):
    corr, p = pearsonr(train_counts, error_rates)
    plt.figure(figsize=(10, 7))
    plt.scatter(train_counts, error_rates, alpha=0.6, color='darkblue')
    z = np.polyfit(train_counts, error_rates, 1)
    plt.plot(train_counts, np.poly1d(z)(train_counts), "r--")
    plt.savefig(output_path)
    plt.close()
