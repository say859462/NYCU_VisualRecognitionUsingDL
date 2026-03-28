import argparse
import json
import os
from collections import Counter

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import ImageDataset
from model import ImageClassificationModel
from utils import PadToSquare


def compute_accuracy(preds, labels):
    if len(labels) == 0:
        return 0.0
    correct = sum(int(p == y) for p, y in zip(preds, labels))
    return 100.0 * correct / len(labels)


def safe_top2_gap(prob_row: torch.Tensor) -> float:
    if prob_row.numel() < 2:
        return 0.0
    top2_prob, _ = torch.topk(prob_row, k=2, dim=0)
    return float((top2_prob[0] - top2_prob[1]).item())


def build_attribution_tag(g_ok, p2_ok, p4_ok, c_ok):
    if g_ok and p2_ok and p4_ok and c_ok:
        return "all_correct"

    if (not g_ok) and (not p2_ok) and (not p4_ok) and (not c_ok):
        return "all_wrong"

    if (not g_ok) and c_ok:
        return "global_wrong_concat_right"

    if (not p2_ok) and c_ok:
        return "part2_wrong_concat_right"

    if (not p4_ok) and c_ok:
        return "part4_wrong_concat_right"

    if p4_ok and (not c_ok):
        return "part4_right_concat_wrong"

    if p2_ok and (not c_ok):
        return "part2_right_concat_wrong"

    if g_ok and (not c_ok):
        return "global_right_concat_wrong"

    if g_ok and (not p2_ok) and (not p4_ok):
        return "only_global_right"

    if (not g_ok) and p2_ok and (not p4_ok):
        return "only_part2_right"

    if (not g_ok) and (not p2_ok) and p4_ok:
        return "only_part4_right"

    if g_ok and p2_ok and (not p4_ok) and (not c_ok):
        return "global_part2_right_concat_wrong"

    if g_ok and (not p2_ok) and p4_ok and (not c_ok):
        return "global_part4_right_concat_wrong"

    if (not g_ok) and p2_ok and p4_ok and (not c_ok):
        return "part2_part4_right_concat_wrong"

    return "mixed"


def main():
    parser = argparse.ArgumentParser(
        description="Detailed PMG analysis with concat output"
    )
    parser.add_argument("--config", type=str, default="./config.json")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--save_dir", type=str,
                        default="./Plot/Analysis_PurePMG_AvgMax")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = args.model_path if args.model_path is not None else config["best_model_path"]

    val_transform = transforms.Compose([
        PadToSquare(fill=(0, 0, 0)),
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    val_dataset = ImageDataset(
        root_dir=config["data_dir"],
        split="val",
        transform=val_transform,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = ImageClassificationModel(
        num_classes=config["num_classes"],
        pretrained=False,
        num_subcenters=config.get("num_subcenters", 3),
        embed_dim=config.get("embed_dim", 256),
    ).to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    rows = []

    all_labels = []
    global_preds = []
    part2_preds = []
    part4_preds = []
    concat_preds = []

    attribution_counter = Counter()
    confusion_counter = Counter()

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Analyzing"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model.forward_pmg(images)

            global_logits = outputs["global_logits"]
            part2_logits = outputs["part2_logits"]
            part4_logits = outputs["part4_logits"]
            concat_logits = outputs["concat_logits"]

            global_prob = torch.softmax(global_logits, dim=1)
            part2_prob = torch.softmax(part2_logits, dim=1)
            part4_prob = torch.softmax(part4_logits, dim=1)
            concat_prob = torch.softmax(concat_logits, dim=1)

            global_pred = torch.argmax(global_prob, dim=1)
            part2_pred = torch.argmax(part2_prob, dim=1)
            part4_pred = torch.argmax(part4_prob, dim=1)
            concat_pred = torch.argmax(concat_prob, dim=1)

            global_preds.extend(global_pred.cpu().tolist())
            part2_preds.extend(part2_pred.cpu().tolist())
            part4_preds.extend(part4_pred.cpu().tolist())
            concat_preds.extend(concat_pred.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            for i in range(labels.size(0)):
                y = labels[i].item()
                gp = global_pred[i].item()
                p2 = part2_pred[i].item()
                p4 = part4_pred[i].item()
                cp = concat_pred[i].item()

                g_ok = int(gp == y)
                p2_ok = int(p2 == y)
                p4_ok = int(p4 == y)
                c_ok = int(cp == y)

                g_conf = float(global_prob[i, gp].item())
                p2_conf = float(part2_prob[i, p2].item())
                p4_conf = float(part4_prob[i, p4].item())
                c_conf = float(concat_prob[i, cp].item())

                g_gap = safe_top2_gap(global_prob[i])
                p2_gap = safe_top2_gap(part2_prob[i])
                p4_gap = safe_top2_gap(part4_prob[i])
                c_gap = safe_top2_gap(concat_prob[i])

                tag = build_attribution_tag(g_ok, p2_ok, p4_ok, c_ok)
                attribution_counter[tag] += 1

                if cp != y:
                    confusion_counter[(y, cp)] += 1

                rows.append({
                    "true_label": y,
                    "global_pred": gp,
                    "part2_pred": p2,
                    "part4_pred": p4,
                    "concat_pred": cp,

                    "global_correct": g_ok,
                    "part2_correct": p2_ok,
                    "part4_correct": p4_ok,
                    "concat_correct": c_ok,

                    "global_conf": g_conf,
                    "part2_conf": p2_conf,
                    "part4_conf": p4_conf,
                    "concat_conf": c_conf,

                    "global_top2_gap": g_gap,
                    "part2_top2_gap": p2_gap,
                    "part4_top2_gap": p4_gap,
                    "concat_top2_gap": c_gap,

                    "all_wrong": int((not g_ok) and (not p2_ok) and (not p4_ok) and (not c_ok)),
                    "all_wrong_high_conf_08": int((not g_ok) and (not p2_ok) and (not p4_ok) and (not c_ok) and (c_conf >= 0.8)),
                    "all_wrong_high_conf_09": int((not g_ok) and (not p2_ok) and (not p4_ok) and (not c_ok) and (c_conf >= 0.9)),
                    "attribution_tag": tag,
                })

    df = pd.DataFrame(rows)

    detailed_csv_path = os.path.join(
        args.save_dir, "pmg_predictions_detailed.csv")
    df.to_csv(detailed_csv_path, index=False)

    concat_error_df = df[df["concat_correct"] == 0]

    summary = {
        "num_samples": len(all_labels),
        "global_acc": compute_accuracy(global_preds, all_labels),
        "part2_acc": compute_accuracy(part2_preds, all_labels),
        "part4_acc": compute_accuracy(part4_preds, all_labels),
        "concat_acc": compute_accuracy(concat_preds, all_labels),

        "case_global_wrong_concat_right": int(((df["global_correct"] == 0) & (df["concat_correct"] == 1)).sum()),
        "case_global_wrong_part2_right": int(((df["global_correct"] == 0) & (df["part2_correct"] == 1)).sum()),
        "case_global_wrong_part4_right": int(((df["global_correct"] == 0) & (df["part4_correct"] == 1)).sum()),

        "case_part2_right_concat_wrong": int(((df["part2_correct"] == 1) & (df["concat_correct"] == 0)).sum()),
        "case_part4_right_concat_wrong": int(((df["part4_correct"] == 1) & (df["concat_correct"] == 0)).sum()),
        "case_global_right_concat_wrong": int(((df["global_correct"] == 1) & (df["concat_correct"] == 0)).sum()),
        "case_any_branch_right_concat_wrong": int((
            ((df["global_correct"] == 1) |
             (df["part2_correct"] == 1) | (df["part4_correct"] == 1))
            & (df["concat_correct"] == 0)
        ).sum()),

        "all_wrong_count": int(df["all_wrong"].sum()),
        "all_wrong_high_conf_08": int(df["all_wrong_high_conf_08"].sum()),
        "all_wrong_high_conf_09": int(df["all_wrong_high_conf_09"].sum()),

        "mean_concat_error_conf": float(concat_error_df["concat_conf"].mean()) if len(concat_error_df) > 0 else 0.0,
        "median_concat_error_conf": float(concat_error_df["concat_conf"].median()) if len(concat_error_df) > 0 else 0.0,
        "mean_concat_error_top2_gap": float(concat_error_df["concat_top2_gap"].mean()) if len(concat_error_df) > 0 else 0.0,
        "high_conf_wrong_count_ge_0.9": int((concat_error_df["concat_conf"] >= 0.9).sum()),
        "high_conf_wrong_count_ge_0.8": int((concat_error_df["concat_conf"] >= 0.8).sum()),
    }

    summary_path = os.path.join(args.save_dir, "analysis_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    branch_attribution_summary = {
        "attribution_tag_counts": dict(attribution_counter),
        "case_global_wrong_concat_right": summary["case_global_wrong_concat_right"],
        "case_global_wrong_part2_right": summary["case_global_wrong_part2_right"],
        "case_global_wrong_part4_right": summary["case_global_wrong_part4_right"],
        "case_part2_right_concat_wrong": summary["case_part2_right_concat_wrong"],
        "case_part4_right_concat_wrong": summary["case_part4_right_concat_wrong"],
        "case_global_right_concat_wrong": summary["case_global_right_concat_wrong"],
        "case_any_branch_right_concat_wrong": summary["case_any_branch_right_concat_wrong"],
        "all_wrong_count": summary["all_wrong_count"],
        "all_wrong_high_conf_08": summary["all_wrong_high_conf_08"],
        "all_wrong_high_conf_09": summary["all_wrong_high_conf_09"],
    }

    branch_summary_path = os.path.join(
        args.save_dir, "branch_attribution_summary.json")
    with open(branch_summary_path, "w", encoding="utf-8") as f:
        json.dump(branch_attribution_summary, f, indent=2, ensure_ascii=False)

    per_class_rows = []
    for cls in sorted(df["true_label"].unique().tolist()):
        cls_df = df[df["true_label"] == cls]
        per_class_rows.append({
            "class_id": int(cls),
            "num_samples": int(len(cls_df)),
            "global_acc": float(100.0 * cls_df["global_correct"].mean()),
            "part2_acc": float(100.0 * cls_df["part2_correct"].mean()),
            "part4_acc": float(100.0 * cls_df["part4_correct"].mean()),
            "concat_acc": float(100.0 * cls_df["concat_correct"].mean()),
            "global_wrong_part4_right": int(((cls_df["global_correct"] == 0) & (cls_df["part4_correct"] == 1)).sum()),
            "part4_right_concat_wrong": int(((cls_df["part4_correct"] == 1) & (cls_df["concat_correct"] == 0)).sum()),
            "all_wrong_count": int(cls_df["all_wrong"].sum()),
        })

    per_class_df = pd.DataFrame(per_class_rows).sort_values(
        by=["concat_acc", "part4_acc", "class_id"], ascending=[True, True, True]
    )
    per_class_csv_path = os.path.join(
        args.save_dir, "per_class_branch_acc.csv")
    per_class_df.to_csv(per_class_csv_path, index=False)

    confusion_rows = []
    for (gt, pred), cnt in confusion_counter.most_common():
        pair_df = df[(df["true_label"] == gt) & (df["concat_pred"] == pred)]
        confusion_rows.append({
            "true_label": int(gt),
            "concat_pred": int(pred),
            "count": int(cnt),
            "global_acc_on_pair_samples": float(100.0 * pair_df["global_correct"].mean()) if len(pair_df) > 0 else 0.0,
            "part2_acc_on_pair_samples": float(100.0 * pair_df["part2_correct"].mean()) if len(pair_df) > 0 else 0.0,
            "part4_acc_on_pair_samples": float(100.0 * pair_df["part4_correct"].mean()) if len(pair_df) > 0 else 0.0,
            "concat_acc_on_pair_samples": float(100.0 * pair_df["concat_correct"].mean()) if len(pair_df) > 0 else 0.0,
            "mean_concat_conf_on_pair_samples": float(pair_df["concat_conf"].mean()) if len(pair_df) > 0 else 0.0,
        })

    confusion_df = pd.DataFrame(confusion_rows)
    if len(confusion_df) > 0:
        confusion_df = confusion_df.sort_values(by="count", ascending=False)

    confusion_csv_path = os.path.join(args.save_dir, "top_confusion_pairs.csv")
    confusion_df.to_csv(confusion_csv_path, index=False)

    print(f"\n===== Analysis Summary ({model_path}) =====")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print("\nSaved files:")
    print(f"- {detailed_csv_path}")
    print(f"- {summary_path}")
    print(f"- {branch_summary_path}")
    print(f"- {per_class_csv_path}")
    print(f"- {confusion_csv_path}")


if __name__ == "__main__":
    main()
