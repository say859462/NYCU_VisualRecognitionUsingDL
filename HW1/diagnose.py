import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

# 載入您的自定義模組
from dataset import ImageDataset
from model import ImageClassificationModel


def get_features_and_preds(model, dataloader, device):
    """
    使用 PyTorch Hook 安全提取 512 維 Embedding 和預測結果
    適配最新的雙表頭 (Dual-Head) 架構
    """
    model.eval()
    all_embeddings = []
    all_preds = []
    all_labels = []

    # 1. 建立 Hook 攔截特徵 (針對 CBP 或 GeM 分類器的輸入)
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            # 分類器的 input 是一個 tuple，我們取第一個元素 [0]
            activation[name] = input[0].detach()
        return hook

    # 自動判斷模型架構並掛上 Hook
    if hasattr(model, 'classifier_cbp'):
        handle = model.classifier_cbp.register_forward_hook(
            get_activation('embed'))
    else:
        handle = model.classifier.register_forward_hook(
            get_activation('embed'))

    # 2. 執行推論
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting Features"):
            images = images.to(device)

            # 在 eval 模式下，模型會自動回傳融合後的 logits (如 logits_ensemble)
            outputs = model(images)

            # 直接取 argmax 即可 (乘不乘 s=20 都不影響最大值的位置)
            preds = torch.argmax(outputs, dim=1)

            all_embeddings.append(activation['embed'].cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())

    # 移除 hook 以免影響後續記憶體
    handle.remove()

    return (np.concatenate(all_embeddings),
            np.concatenate(all_preds),
            np.concatenate(all_labels))


def plot_global_confusion_matrix(y_true, y_pred, num_classes, save_path):
    """
    繪製完整的 100x100 混淆矩陣
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))

    plt.figure(figsize=(24, 20))  # 尺寸需夠大才塞得下 100 類
    sns.heatmap(cm, cmap='Blues', xticklabels=True,
                yticklabels=True, annot=False)
    plt.title("Global Confusion Matrix (100 Classes)",
              fontsize=24, fontweight='bold')
    plt.ylabel('True Label', fontsize=18)
    plt.xlabel('Predicted Label', fontsize=18)
    # 確保刻度清晰
    plt.xticks(fontsize=6, rotation=90)
    plt.yticks(fontsize=6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📊 Global Confusion Matrix saved to: {save_path}")


def analyze_top_confusions(y_true, y_pred, num_classes, save_csv_path, top_k=20):
    """
    自動抓出最常被混淆的 Top-K 類別對，並輸出為 CSV
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    np.fill_diagonal(cm, 0)  # 將對角線 (預測正確的) 設為 0，只看錯誤的

    confusions = []
    for true_class in range(num_classes):
        for pred_class in range(num_classes):
            count = cm[true_class, pred_class]
            if count > 0:
                confusions.append({
                    'True_Class': true_class,
                    'Pred_Class': pred_class,
                    'Error_Count': count
                })

    df = pd.DataFrame(confusions)
    if not df.empty:
        df = df.sort_values(by='Error_Count', ascending=False).head(top_k)
        df.to_csv(save_csv_path, index=False)
        print(f"⚠️ Top {top_k} Confusions saved to: {save_csv_path}")
        print("\n--- Top Confused Class Pairs ---")
        print(df.to_string(index=False))
        print("--------------------------------\n")
    else:
        print("🎉 No confusions found! (100% accuracy)")


def plot_local_confusion_matrix(y_true, y_pred, target_classes, save_path):
    """
    繪製特定易混淆類別的局部混淆矩陣
    """
    mask = np.isin(y_true, target_classes) & np.isin(y_pred, target_classes)
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]

    if len(y_true_filtered) == 0:
        print(
            f"No predictions found among the target classes {target_classes} to plot local CM.")
        return

    cm = confusion_matrix(
        y_true_filtered, y_pred_filtered, labels=target_classes)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                xticklabels=target_classes, yticklabels=target_classes)
    plt.title(
        f"Local Confusion Matrix\n(Focus Classes: {target_classes})", fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"🎯 Local Confusion Matrix saved to: {save_path}")


def plot_tsne(embeddings, labels, target_classes, save_path):
    """
    繪製 t-SNE (適配最新版 sklearn)
    """
    mask = np.isin(labels, target_classes)
    feat_subset = embeddings[mask]
    label_subset = labels[mask]

    if len(feat_subset) == 0:
        print("No samples found for the target classes for t-SNE.")
        return

    print(f"Running t-SNE on {len(feat_subset)} samples...")
    tsne = TSNE(
        n_components=2,
        perplexity=min(30, len(feat_subset) - 1),
        random_state=42,
        init='pca',
        learning_rate='auto'
    )
    embed_2d = tsne.fit_transform(feat_subset)

    plt.figure(figsize=(12, 10))
    unique_classes = np.unique(label_subset)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))

    for i, class_id in enumerate(unique_classes):
        class_mask = (label_subset == class_id)
        plt.scatter(
            embed_2d[class_mask, 0],
            embed_2d[class_mask, 1],
            label=f"Class {class_id}",
            color=colors[i],
            alpha=0.8,
            s=60,
            edgecolors='w'
        )

    plt.legend(title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f"t-SNE Feature Space Visualization",
              fontsize=16, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"🌌 t-SNE plot saved to: {save_path}")


def main():
    # ==========================================
    # 1. 參數設定
    # ==========================================
    MODEL_PATH = './Model_Weight/best_model.pth'
    DATA_DIR = './Dataset/data'
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IMG_SIZE = 512
    OUTPUT_DIR = './Plot/EXP29_Diagnostic'

    # 預設先看一些可能是蝴蝶或鳥類的 ID，您可以跑完 CSV 後再回來改這組數字
    TARGET_CLASSES = [2, 76, 58, 20, 6, 16, 86,
                      45, 44, 70, 48, 46, 56, 31, 75, 50, 81, 35]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Device: {DEVICE} | Output Dir: {OUTPUT_DIR}")

    # ==========================================
    # 2. 資料準備
    # ==========================================
    val_transform = transforms.Compose([
        transforms.Resize(640),
        transforms.CenterCrop(576),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])
    val_dataset = ImageDataset(
        root_dir=DATA_DIR, split="val", transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=16,
                            shuffle=False, num_workers=4)

    # ==========================================
    # 3. 模型載入與特徵提取
    # ==========================================
    model = ImageClassificationModel(
        num_classes=100, pretrained=False).to(DEVICE)

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"✅ Loaded weights from {MODEL_PATH}")
    else:
        print(f"❌ Error: Model weight not found at {MODEL_PATH}")
        return

    embeddings, preds, labels = get_features_and_preds(
        model, val_loader, DEVICE)

    # ==========================================
    # 4. 全面診斷分析
    # ==========================================
    # (A) 匯出最常混淆的 Top 20 類別 (最強大的除錯工具)
    analyze_top_confusions(labels, preds, num_classes=100,
                           save_csv_path=f'{OUTPUT_DIR}/top_confused_pairs.csv', top_k=20)

    # (B) 繪製 100x100 全局混淆矩陣
    plot_global_confusion_matrix(labels, preds, num_classes=100,
                                 save_path=f'{OUTPUT_DIR}/cm_global.png')

    # (C) 繪製局部混淆矩陣 (針對 Target Classes)
    plot_local_confusion_matrix(labels, preds, TARGET_CLASSES,
                                f'{OUTPUT_DIR}/cm_local_target.png')

    # (D) 繪製 t-SNE 特徵分布圖
    plot_tsne(embeddings, labels, TARGET_CLASSES,
              f'{OUTPUT_DIR}/tsne_local_target.png')


if __name__ == "__main__":
    main()
