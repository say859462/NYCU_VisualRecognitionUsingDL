import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import torchvision.transforms.functional as TF

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=1.0):
        """
        LDAM Loss (Label-Distribution-Aware Margin Loss)
        :param cls_num_list: 每個類別的樣本數量列表 (list or numpy array)
        :param max_m: 最大 Margin 值 (預設 0.5)
        :param weight: 類別權重 (DRW 策略會在第二階段傳入 CB_Weights)
        :param s: 縮放因子 (若您的 Classifier 是標準 nn.Linear，建議設 1.0；若是 Cosine 分類器，通常設 30)
        """
        super(LDAMLoss, self).__init__()
        # 核心公式：Margin 與樣本數的四次方根成反比
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        self.m_list = torch.FloatTensor(m_list)
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        # 建立與 logits (x) 大小相同的 boolean mask
        index = torch.zeros_like(x, dtype=torch.bool)
        index.scatter_(1, target.data.view(-1, 1), 1)

        # 取得每個 batch 樣本對應的 margin
        index_float = index.type(torch.FloatTensor).to(x.device)
        batch_m = torch.matmul(self.m_list[None, :].to(
            x.device), index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))

        # 將正確類別的 logit 減去 margin (強迫模型拉開安全距離)
        x_m = x - batch_m

        # 將減去 margin 的 logit 放回原本的張量中
        output = torch.where(index, x_m, x)

        # 套用縮放與權重，並計算 CrossEntropy
        return F.cross_entropy(self.s * output, target, weight=self.weight)


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
    plt.xlabel("Class ID", fontsize=12)  # ⭐ 加 X 軸標籤
    plt.ylabel("Number of Images", fontsize=12)  # ⭐ 加 Y 軸標籤
    plt.xticks(range(len(counts)), rotation=90, fontsize=8)  # ⭐ 顯示所有 X 座標
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

    # --- Loss Plot (左圖) ---
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch', fontsize=12)  # 保留座標標籤
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend()
    ax1.set_title('Training and Validation Loss', fontsize=14)

    # --- Accuracy Plot (右圖) ---
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

    plt.figure(figsize=(20, 6))  # 加寬圖片以容納標籤
    plt.bar(range(num_classes), error_rates, color='salmon', edgecolor='black')
    plt.axhline(y=np.mean(error_rates), color='red', linestyle='--',
                label=f'Mean Error ({np.mean(error_rates):.1f}%)')

    plt.xlabel('Class ID', fontsize=12)  # ⭐ 加 X 軸標籤
    plt.ylabel('Error Rate (%)', fontsize=12)  # ⭐ 加 Y 軸標籤
    plt.xticks(range(num_classes), rotation=90,
               fontsize=8)  # ⭐ 顯示出 0~99 所有 Class ID
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
    ax1.set_xlabel('Class ID (Sorted by Image Count)',
                   fontsize=12)  # ⭐ 加 X 軸標籤
    ax1.set_ylabel('Number of Training Images',
                   color='skyblue', fontsize=12)  # ⭐ 加 Y 軸標籤
    ax1.set_xticks(range(num_classes))
    ax1.set_xticklabels(sorted_idx, rotation=90,
                        fontsize=7)  # ⭐ 顯示排序後的 Class ID

    ax2 = ax1.twinx()
    ax2.plot(range(num_classes), val_accs[sorted_idx],
             color='red', marker='o', markersize=4, label='Val Accuracy')
    ax2.set_ylabel('Validation Accuracy (%)',
                   color='red', fontsize=12)  # ⭐ 加 Y 軸標籤

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
    plt.xlabel('Number of Training Images', fontsize=12)  # ⭐ 加 X 軸標籤
    plt.ylabel('Error Rate (%)', fontsize=12)  # ⭐ 加 Y 軸標籤
    plt.title(f'Correlation Analysis (Pearson r: {corr:.4f})', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
