import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

from dataset import ImageDataset
from model import ImageClassificationModel


def get_embeddings_preds_labels_paths(model, dataloader, device, image_paths):
    model.eval()

    all_embeddings = []
    all_logits = []
    all_logits_all = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting Features", colour="cyan"):
            images = images.to(device)
            labels = labels.to(device)

            pooled, _ = model.forward_features(images)
            logits, embed, logits_all = model.forward_head(pooled)

            preds = torch.argmax(logits, dim=1)

            all_embeddings.append(embed.cpu().numpy())
            all_logits.append(logits.cpu().numpy())
            all_logits_all.append(logits_all.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    logits = np.concatenate(all_logits, axis=0)
    logits_all = np.concatenate(all_logits_all, axis=0)
    preds = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # ✅ safer softmax
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    confs = probs.max(axis=1)

    # ===============================
    # ✅ 正確計算 subcenter
    # ===============================
    assigned_subcenters = []

    for i in range(len(preds)):
        pred_cls = preds[i]

        # logits_all: [C, K]
        sub_id = int(np.argmax(logits_all[i, pred_cls]))
        assigned_subcenters.append(sub_id)

    assigned_subcenters = np.array(assigned_subcenters)

    return (
        embeddings,
        logits,
        logits_all,
        probs,
        preds,
        labels,
        confs,
        assigned_subcenters,
        image_paths
    )


def plot_global_confusion_matrix(y_true, y_pred, num_classes, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))

    plt.figure(figsize=(24, 20))
    sns.heatmap(cm, cmap='Blues', xticklabels=True,
                yticklabels=True, annot=False)
    plt.title("Global Confusion Matrix (100 Classes)",
              fontsize=22, fontweight='bold')
    plt.ylabel("True Label", fontsize=16)
    plt.xlabel("Predicted Label", fontsize=16)
    plt.xticks(fontsize=6, rotation=90)
    plt.yticks(fontsize=6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📊 Saved global confusion matrix -> {save_path}")


def analyze_top_confusions(y_true, y_pred, num_classes, save_csv_path, top_k=20):
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    np.fill_diagonal(cm, 0)

    rows = []
    for true_class in range(num_classes):
        for pred_class in range(num_classes):
            count = cm[true_class, pred_class]
            if count > 0:
                rows.append({
                    "True_Class": true_class,
                    "Pred_Class": pred_class,
                    "Error_Count": int(count)
                })

    df = pd.DataFrame(rows)
    if len(df) == 0:
        print("🎉 No confusions found.")
        return df

    df = df.sort_values(by="Error_Count", ascending=False).head(
        top_k).reset_index(drop=True)
    df.to_csv(save_csv_path, index=False)
    print(f"⚠️ Saved top confused pairs -> {save_csv_path}")
    print(df.to_string(index=False))
    return df


def build_per_class_summary(train_labels, val_labels, val_preds, num_classes, save_csv_path):
    train_counts = np.bincount(train_labels, minlength=num_classes)
    val_counts = np.bincount(val_labels, minlength=num_classes)

    cm = confusion_matrix(val_labels, val_preds, labels=range(num_classes))
    summary = []

    for cls in range(num_classes):
        correct = int(cm[cls, cls])
        total = int(val_counts[cls])
        val_acc = (correct / total * 100.0) if total > 0 else 0.0

        row = cm[cls].copy()
        row[cls] = 0
        top_mistaken_class = int(np.argmax(row)) if row.sum() > 0 else -1
        top_mistake_count = int(
            row[top_mistaken_class]) if row.sum() > 0 else 0

        summary.append({
            "Class_ID": cls,
            "Train_Count": int(train_counts[cls]),
            "Val_Count": total,
            "Val_Correct": correct,
            "Val_Acc(%)": round(val_acc, 2),
            "Top_Mistaken_As": top_mistaken_class,
            "Top_Mistake_Count": top_mistake_count
        })

    df = pd.DataFrame(summary)
    df = df.sort_values(by=["Val_Acc(%)", "Train_Count"], ascending=[
                        True, True]).reset_index(drop=True)
    df.to_csv(save_csv_path, index=False)
    print(f"📄 Saved per-class summary -> {save_csv_path}")
    return df


def export_hardest_mistakes(image_paths, labels, preds, confs, probs, save_csv_path, top_k=100):
    rows = []
    for path, y, p, conf, prob_vec in zip(image_paths, labels, preds, confs, probs):
        if y != p:
            rows.append({
                "Image_Path": path,
                "True_Class": int(y),
                "Pred_Class": int(p),
                "Pred_Conf": float(conf),
                "True_Class_Prob": float(prob_vec[y]),
            })

    df = pd.DataFrame(rows)
    if len(df) == 0:
        print("🎉 No mistakes found.")
        return df

    df = df.sort_values(by="Pred_Conf", ascending=False).head(
        top_k).reset_index(drop=True)
    df.to_csv(save_csv_path, index=False)
    print(f"🧨 Saved hardest mistakes -> {save_csv_path}")
    return df


def export_subcenter_assignments(
    image_paths,
    labels,
    preds,
    confs,
    assigned_subcenters,
    save_csv_path
):
    rows = []
    for path, y, p, conf, sub in zip(
        image_paths, labels, preds, confs, assigned_subcenters
    ):
        rows.append({
            "Image_Path": path,
            "True_Class": int(y),
            "Pred_Class": int(p),
            "Pred_Confidence": float(conf),
            "Assigned_Subcenter": int(sub)
        })

    df = pd.DataFrame(rows)
    df.to_csv(save_csv_path, index=False)
    print(f"🧩 Saved subcenter assignments -> {save_csv_path}")
    return df


def build_subcenter_summary(labels, preds, assigned_subcenters, save_csv_path):
    """
    只看 prediction correct 的樣本，
    分析每個 class 內部不同 subcenter 各自吸了多少資料。
    """
    rows = []

    labels = np.array(labels)
    preds = np.array(preds)
    assigned_subcenters = np.array(assigned_subcenters)

    correct_mask = (labels == preds)

    unique_classes = np.unique(labels)
    for cls in unique_classes:
        cls_mask = (labels == cls) & correct_mask
        if cls_mask.sum() == 0:
            continue

        sub_ids, counts = np.unique(
            assigned_subcenters[cls_mask], return_counts=True)
        total = int(cls_mask.sum())

        for sub_id, cnt in zip(sub_ids, counts):
            rows.append({
                "Class_ID": int(cls),
                "Subcenter_ID": int(sub_id),
                "Correct_Assigned_Count": int(cnt),
                "Ratio_in_Class(%)": round(cnt / total * 100.0, 2)
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(
        by=["Class_ID", "Correct_Assigned_Count"], ascending=[True, False])
    df.to_csv(save_csv_path, index=False)
    print(f"📄 Saved subcenter summary -> {save_csv_path}")
    return df


def get_top_confused_class_set(confused_df, max_classes=12):
    if confused_df is None or len(confused_df) == 0:
        return []

    classes = []
    for _, row in confused_df.iterrows():
        classes.extend([int(row["True_Class"]), int(row["Pred_Class"])])

    counts = pd.Series(classes).value_counts()
    return counts.head(max_classes).index.tolist()


def plot_local_confusion_matrix(y_true, y_pred, target_classes, save_path):
    if len(target_classes) == 0:
        print("No target classes for local confusion matrix.")
        return

    mask = np.isin(y_true, target_classes) & np.isin(y_pred, target_classes)
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]

    if len(y_true_filtered) == 0:
        print(
            "No samples available among selected target classes for local confusion matrix.")
        return

    cm = confusion_matrix(
        y_true_filtered, y_pred_filtered, labels=target_classes)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Reds',
        xticklabels=target_classes,
        yticklabels=target_classes
    )
    plt.title("Local Confusion Matrix (Top Confused Classes)", fontweight='bold')
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"🎯 Saved local confusion matrix -> {save_path}")


def plot_tsne(embeddings, labels, target_classes, save_path):
    if len(target_classes) == 0:
        print("No target classes for t-SNE.")
        return

    mask = np.isin(labels, target_classes)
    feat_subset = embeddings[mask]
    label_subset = labels[mask]

    if len(feat_subset) < 5:
        print("Not enough samples for t-SNE.")
        return

    perplexity = min(30, max(5, len(feat_subset) - 1))
    print(f"Running t-SNE on {len(feat_subset)} samples...")

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42,
        init='pca',
        learning_rate='auto'
    )
    embed_2d = tsne.fit_transform(feat_subset)

    plt.figure(figsize=(12, 10))
    unique_classes = np.unique(label_subset)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_classes)))

    for i, class_id in enumerate(unique_classes):
        class_mask = (label_subset == class_id)
        plt.scatter(
            embed_2d[class_mask, 0],
            embed_2d[class_mask, 1],
            label=f"Class {class_id}",
            color=colors[i],
            alpha=0.8,
            s=45,
            edgecolors='w'
        )

    plt.legend(title="Classes", bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.title("t-SNE of Top Confused Classes", fontsize=16, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"🌌 Saved t-SNE -> {save_path}")


def plot_tsne_by_subcenter(embeddings, labels, assigned_subcenters, target_class, save_path):
    """
    只看單一 class，依 subcenter 著色
    """
    mask = (labels == target_class)
    feat_subset = embeddings[mask]
    sub_subset = assigned_subcenters[mask]

    if len(feat_subset) < 5:
        print(f"Not enough samples for class {target_class} t-SNE.")
        return

    perplexity = min(30, max(5, len(feat_subset) - 1))
    print(
        f"Running class-{target_class} subcenter t-SNE on {len(feat_subset)} samples...")

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42,
        init='pca',
        learning_rate='auto'
    )
    embed_2d = tsne.fit_transform(feat_subset)

    plt.figure(figsize=(10, 8))
    unique_subs = np.unique(sub_subset)
    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_subs)))

    for i, sub_id in enumerate(unique_subs):
        sub_mask = (sub_subset == sub_id)
        plt.scatter(
            embed_2d[sub_mask, 0],
            embed_2d[sub_mask, 1],
            label=f"Subcenter {sub_id}",
            color=colors[i],
            alpha=0.8,
            s=55,
            edgecolors='w'
        )

    plt.legend()
    plt.title(f"t-SNE of Class {target_class} by Subcenter",
              fontsize=15, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"🌌 Saved class-{target_class} subcenter t-SNE -> {save_path}")


def main():
    MODEL_PATH = "./Model_Weight/best_model.pth"
    DATA_DIR = "./Dataset/data"
    OUTPUT_DIR = "./Plot/Subcenter_Diagnostic_56th"
    NUM_CLASSES = 100
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 你可以手動指定要觀察的 class
    TARGET_SUBCENTER_TSNE_CLASSES = [2, 6, 20, 44, 45, 76, 88]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Device: {DEVICE}")
    print(f"Output Dir: {OUTPUT_DIR}")

    val_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_dataset = ImageDataset(
        root_dir=DATA_DIR, split="val", transform=val_transform)
    train_dataset = ImageDataset(
        root_dir=DATA_DIR, split="train", transform=None)

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4
    )

    model = ImageClassificationModel(
        num_classes=NUM_CLASSES,
        pretrained=False,
        num_subcenters=3,
        embed_dim=256
    ).to(DEVICE)

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model weight not found: {MODEL_PATH}")
        return

    state = torch.load(MODEL_PATH, map_location=DEVICE)

    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
        print("✅ Loaded model weights from state['model']")
    else:
        model.load_state_dict(state)
        print("✅ Loaded raw state_dict")
    print(f"✅ Loaded model from: {MODEL_PATH}")

    (
        embeddings,
        logits,
        logits_all,
        probs,
        preds,
        labels,
        confs,
        assigned_subcenters,
        image_paths
    ) = get_embeddings_preds_labels_paths(
        model,
        val_loader,
        DEVICE,
        val_dataset.image_paths
    )

    # 1) top confused pairs
    confused_df = analyze_top_confusions(
        labels,
        preds,
        num_classes=NUM_CLASSES,
        save_csv_path=os.path.join(OUTPUT_DIR, "top_confused_pairs.csv"),
        top_k=20
    )

    # 2) per-class summary
    per_class_df = build_per_class_summary(
        train_labels=np.array(train_dataset.targets),
        val_labels=labels,
        val_preds=preds,
        num_classes=NUM_CLASSES,
        save_csv_path=os.path.join(OUTPUT_DIR, "per_class_summary.csv")
    )

    # 3) hardest mistakes
    hardest_df = export_hardest_mistakes(
        image_paths=image_paths,
        labels=labels,
        preds=preds,
        confs=confs,
        probs=probs,
        save_csv_path=os.path.join(OUTPUT_DIR, "hardest_mistakes.csv"),
        top_k=100
    )

    # 4) 每張圖的 subcenter assignment
    subcenter_assign_df = export_subcenter_assignments(
        image_paths=image_paths,
        labels=labels,
        preds=preds,
        confs=confs,
        assigned_subcenters=assigned_subcenters,
        save_csv_path=os.path.join(OUTPUT_DIR, "subcenter_assignments.csv")
    )

    # 5) 每個 class 內，各 subcenter 吸了多少正確樣本
    subcenter_summary_df = build_subcenter_summary(
        labels=labels,
        preds=preds,
        assigned_subcenters=assigned_subcenters,
        save_csv_path=os.path.join(OUTPUT_DIR, "subcenter_summary.csv")
    )

    # 6) global confusion matrix
    plot_global_confusion_matrix(
        labels,
        preds,
        num_classes=NUM_CLASSES,
        save_path=os.path.join(OUTPUT_DIR, "cm_global.png")
    )

    # 7) auto target classes from confused pairs
    target_classes = get_top_confused_class_set(confused_df, max_classes=12)
    print(f"🎯 Auto selected target classes: {target_classes}")

    # 8) local confusion matrix
    plot_local_confusion_matrix(
        labels,
        preds,
        target_classes=target_classes,
        save_path=os.path.join(OUTPUT_DIR, "cm_top_confused_classes.png")
    )

    # 9) t-SNE of confused classes
    plot_tsne(
        embeddings,
        labels,
        target_classes=target_classes,
        save_path=os.path.join(OUTPUT_DIR, "tsne_top_confused_classes.png")
    )

    # 10) class-wise t-SNE by subcenter
    for cls in TARGET_SUBCENTER_TSNE_CLASSES:
        plot_tsne_by_subcenter(
            embeddings=embeddings,
            labels=labels,
            assigned_subcenters=assigned_subcenters,
            target_class=cls,
            save_path=os.path.join(
                OUTPUT_DIR, f"tsne_class_{cls}_subcenters.png")
        )

    print("\n✅ Analysis complete.")
    print("Generated files:")
    print("- top_confused_pairs.csv")
    print("- per_class_summary.csv")
    print("- hardest_mistakes.csv")
    print("- subcenter_assignments.csv")
    print("- subcenter_summary.csv")
    print("- cm_global.png")
    print("- cm_top_confused_classes.png")
    print("- tsne_top_confused_classes.png")
    print("- tsne_class_<id>_subcenters.png")


if __name__ == "__main__":
    main()
