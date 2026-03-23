import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import torchvision.transforms.functional as TF
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CurricularFace(nn.Module):
    """CurricularFace: 動態 Margin 策略，解決初期梯度崩潰與長尾困難樣本問題"""
    def __init__(self, s=30.0, m=0.5):
        super(CurricularFace, self).__init__()
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.register_buffer('t', torch.zeros(1))

    def forward(self, cosine, label):
        cosine = cosine.float()
        cosine = torch.clamp(cosine, -1.0 + 1e-5, 1.0 - 1e-5)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        with torch.no_grad():
            alpha = 0.99
            pos_cos = torch.masked_select(cosine, one_hot.bool())
            pos_mean = torch.mean(pos_cos) if pos_cos.numel() > 0 else self.t
            self.t = alpha * self.t + (1 - alpha) * pos_mean

        # 針對大於 t 的負樣本施加動態懲罰
        neg_cosine = torch.where(cosine > self.t, cosine * (self.t + cosine), cosine)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * neg_cosine)
        return F.cross_entropy(output * self.s, label)
class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss (單視角批次內版本)
    專門解決極相似物種 (FGVC) 的特徵推擠，不設定硬性 Margin，而是透過溫度係數平滑拉開邊界。
    """

    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # L2 正規化
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)

        # 計算特徵相似度點積矩陣
        anchor_dot_contrast = torch.div(torch.matmul(
            features, features.T), self.temperature)

        # 數值穩定性處理
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # 屏蔽對角線 (自己與自己的相似度不納入計算)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(features.device),
            0
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        mask_sum = mask.sum(1)
        mask_sum = torch.where(
            mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

        return - mean_log_prob_pos.mean()


class ArcFaceLoss(nn.Module):
    def __init__(self, s=30.0, m=0.5, weight=None, label_smooth=0.05):
        """
        ArcFace Loss: 直接在餘弦空間施加角度 Margin，與 Cosine Classifier 完美契合。
        :param s: 縮放因子 (Scale)，FGVC 通常設定 30.0
        :param m: 角度邊距 (Angular Margin)，通常設定 0.5
        :param weight: 類別權重 (用於 DRW 階段)
        """
        super(ArcFaceLoss, self).__init__()
        self.s = s
        self.m = m
        self.weight = weight
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.label_smooth = label_smooth
        # 安全閾值，防止數值不穩定
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, cosine, label):
        # 確保 cosine 在安全範圍內，避免 sqrt 產生 NaN
        cosine = cosine.float()

        # 確保 cosine 在安全範圍內，改用 1e-5 以確保絕對安全
        cosine = torch.clamp(cosine, -1.0 + 1e-5, 1.0 - 1e-5)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))

        # cos(θ + m) = cosθ * cosm - sinθ * sinm
        phi = cosine * self.cos_m - sine * self.sin_m

        # 數值穩定性處理 (如果角度已經太大，則線性遞減)
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # 將 Margin 只加在正確答案 (Ground Truth) 對應的 Logit 上
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        # 乘上縮放因子 s，然後計算 Cross Entropy
        return F.cross_entropy(output * self.s, label, weight=self.weight, label_smoothing=self.label_smooth)


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
