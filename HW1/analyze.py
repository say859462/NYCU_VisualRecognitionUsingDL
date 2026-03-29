import argparse
import json
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import ImageDataset
from model import ImageClassificationModel


def compute_accuracy(preds, labels):
    if len(labels) == 0:
        return 0.0
    correct = sum(int(pred == label) for pred, label in zip(preds, labels))
    return 100.0 * correct / len(labels)


def safe_top2_gap(prob_row: torch.Tensor) -> float:
    if prob_row.numel() < 2:
        return 0.0
    top2_prob, _ = torch.topk(prob_row, k=2, dim=0)
    return float((top2_prob[0] - top2_prob[1]).item())


def build_per_class_stats(df: pd.DataFrame, num_classes: int):
    rows = []
    for class_id in range(num_classes):
        class_df = df[df["true_label"] == class_id]
        if len(class_df) == 0:
            rows.append({
                "class_id": class_id,
                "num_samples": 0,
                "global_acc": 0.0,
                "part2_acc": 0.0,
                "part4_acc": 0.0,
                "concat_acc": 0.0,
                "mean_concat_conf": 0.0,
                "mean_concat_top2_gap": 0.0,
            })
            continue

        row = {
            "class_id": class_id,
            "num_samples": int(len(class_df)),
            "global_acc": float(class_df["global_correct"].mean() * 100.0),
            "part2_acc": float(class_df["part2_correct"].mean() * 100.0),
            "part4_acc": float(class_df["part4_correct"].mean() * 100.0),
            "concat_acc": float(class_df["concat_correct"].mean() * 100.0),
            "mean_concat_conf": float(class_df["concat_conf"].mean()),
            "mean_concat_top2_gap": float(class_df["concat_top2_gap"].mean()),
        }

        optional_cols = [
            "cls_attn_to_global",
            "cls_attn_to_part2_group_sum",
            "cls_attn_to_part4_group_sum",
            "cls_attn_to_part2_group_max",
            "cls_attn_to_part4_group_max",
            "mean_selected_part4_score",
        ]
        for col in optional_cols:
            if col in class_df.columns:
                row[f"mean_{col}"] = float(class_df[col].mean())

        rows.append(row)

    return pd.DataFrame(rows)


def dominant_source_from_scores(global_score, part2_score, part4_score):
    scores = {
        "global": global_score,
        "part2_group": part2_score,
        "part4_group": part4_score,
    }
    return max(scores, key=scores.get)


def main():
    parser = argparse.ArgumentParser(
        description="Detailed PMG analysis for Top-k part4 spatial-token fusion"
    )
    parser.add_argument("--config", type=str, default="./config.json")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./Plot/Analysis_ResNet152_PartialRes2Net_TopKSpatialTokenFusion",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--resize", type=int, default=576)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = args.model_path if args.model_path is not None else config["best_model_path"]

    val_transform = transforms.Compose([
        transforms.Resize((args.resize, args.resize)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
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
        use_logit_router=config.get("use_logit_router", False),
        router_hidden_dim=config.get("router_hidden_dim", 256),
        router_dropout=config.get("router_dropout", 0.1),
        backbone_name=config.get("backbone_name", "resnet152_partial_res2net"),
        part4_topk=config.get("part4_topk", 4),
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

    fusion_cls_attn_sum = None
    fusion_cls_attn_count = 0

    dominant_source_counter = {
        "global": 0,
        "part2_group": 0,
        "part4_group": 0,
    }

    selected_part4_hist = torch.zeros(16, dtype=torch.long)

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

            fusion_attn = outputs.get("fusion_attn", None)
            selected_part4_idx = outputs.get("selected_part4_idx", None)
            part4_token_scores = outputs.get("part4_token_scores", None)
            part4_topk_scores = outputs.get("part4_topk_scores", None)

            if selected_part4_idx is not None:
                for i in range(selected_part4_idx.size(0)):
                    for idx in selected_part4_idx[i].detach().cpu().tolist():
                        selected_part4_hist[idx] += 1

            if fusion_attn is not None:
                # token layout after top-k selection:
                # 0: CLS
                # 1: global
                # 2~5: part2 tokens
                # 6~(6+K-1): selected part4 tokens
                cls_attn = fusion_attn[:, 0, :]
                batch_sum = cls_attn.sum(dim=0).detach().cpu()
                if fusion_cls_attn_sum is None:
                    fusion_cls_attn_sum = torch.zeros_like(batch_sum)
                fusion_cls_attn_sum += batch_sum
                fusion_cls_attn_count += cls_attn.size(0)

            for i in range(labels.size(0)):
                y = labels[i].item()
                gp = global_pred[i].item()
                p2 = part2_pred[i].item()
                p4 = part4_pred[i].item()
                cp = concat_pred[i].item()

                row = {
                    "true_label": y,

                    "global_pred": gp,
                    "part2_pred": p2,
                    "part4_pred": p4,
                    "concat_pred": cp,

                    "global_correct": int(gp == y),
                    "part2_correct": int(p2 == y),
                    "part4_correct": int(p4 == y),
                    "concat_correct": int(cp == y),

                    "global_conf": float(global_prob[i, gp].item()),
                    "part2_conf": float(part2_prob[i, p2].item()),
                    "part4_conf": float(part4_prob[i, p4].item()),
                    "concat_conf": float(concat_prob[i, cp].item()),

                    "global_top2_gap": safe_top2_gap(global_prob[i]),
                    "part2_top2_gap": safe_top2_gap(part2_prob[i]),
                    "part4_top2_gap": safe_top2_gap(part4_prob[i]),
                    "concat_top2_gap": safe_top2_gap(concat_prob[i]),
                }

                if selected_part4_idx is not None:
                    idx_list = selected_part4_idx[i].detach().cpu().tolist()
                    row["selected_part4_idx"] = ",".join(map(str, idx_list))

                if part4_token_scores is not None:
                    token_scores_i = part4_token_scores[i].detach().cpu()
                    row["part4_score_mean_all"] = float(
                        token_scores_i.mean().item())
                    row["part4_score_max_all"] = float(
                        token_scores_i.max().item())
                    row["part4_score_argmax_all"] = int(
                        token_scores_i.argmax().item())

                if part4_topk_scores is not None:
                    topk_scores_i = part4_topk_scores[i].detach().cpu()
                    row["mean_selected_part4_score"] = float(
                        topk_scores_i.mean().item())
                    row["max_selected_part4_score"] = float(
                        topk_scores_i.max().item())

                if fusion_attn is not None:
                    cls_attn_i = fusion_attn[i, 0, :].detach().cpu()

                    global_score = float(cls_attn_i[1].item())
                    part2_scores = cls_attn_i[2:6]
                    part4_scores = cls_attn_i[6:]

                    part2_sum = float(part2_scores.sum().item())
                    part4_sum = float(part4_scores.sum().item())

                    part2_mean = float(part2_scores.mean().item(
                    )) if part2_scores.numel() > 0 else 0.0
                    part4_mean = float(part4_scores.mean().item(
                    )) if part4_scores.numel() > 0 else 0.0

                    part2_max = float(part2_scores.max().item()
                                      ) if part2_scores.numel() > 0 else 0.0
                    part4_max = float(part4_scores.max().item()
                                      ) if part4_scores.numel() > 0 else 0.0

                    part2_top_idx = int(part2_scores.argmax(
                    ).item()) if part2_scores.numel() > 0 else -1
                    part4_top_idx = int(part4_scores.argmax(
                    ).item()) if part4_scores.numel() > 0 else -1

                    dominant_source = dominant_source_from_scores(
                        global_score,
                        part2_sum,
                        part4_sum,
                    )
                    dominant_source_counter[dominant_source] += 1

                    row.update({
                        "cls_attn_to_cls": float(cls_attn_i[0].item()),
                        "cls_attn_to_global": global_score,

                        "cls_attn_to_part2_group_sum": part2_sum,
                        "cls_attn_to_part4_group_sum": part4_sum,

                        "cls_attn_to_part2_group_mean": part2_mean,
                        "cls_attn_to_part4_group_mean": part4_mean,

                        "cls_attn_to_part2_group_max": part2_max,
                        "cls_attn_to_part4_group_max": part4_max,

                        "cls_attn_part2_top_token_idx": part2_top_idx,
                        "cls_attn_part4_top_token_idx": part4_top_idx,

                        "cls_attn_dominant_source": dominant_source,
                    })

                rows.append(row)

    df = pd.DataFrame(rows)
    detailed_csv_path = os.path.join(
        args.save_dir, "pmg_predictions_detailed.csv")
    df.to_csv(detailed_csv_path, index=False)

    concat_error_df = df[df["concat_correct"] == 0]

    summary = {
        "backbone_name": config.get("backbone_name", "resnet152_partial_res2net"),
        "resize": args.resize,
        "num_samples": len(all_labels),

        "global_acc": compute_accuracy(global_preds, all_labels),
        "part2_acc": compute_accuracy(part2_preds, all_labels),
        "part4_acc": compute_accuracy(part4_preds, all_labels),
        "concat_acc": compute_accuracy(concat_preds, all_labels),

        "case_global_wrong_concat_right": int(
            ((df["global_correct"] == 0) & (df["concat_correct"] == 1)).sum()
        ),
        "case_part2_wrong_concat_right": int(
            ((df["part2_correct"] == 0) & (df["concat_correct"] == 1)).sum()
        ),
        "case_part4_wrong_concat_right": int(
            ((df["part4_correct"] == 0) & (df["concat_correct"] == 1)).sum()
        ),

        "case_part2_right_concat_wrong": int(
            ((df["part2_correct"] == 1) & (df["concat_correct"] == 0)).sum()
        ),
        "case_part4_right_concat_wrong": int(
            ((df["part4_correct"] == 1) & (df["concat_correct"] == 0)).sum()
        ),
        "case_global_right_concat_wrong": int(
            ((df["global_correct"] == 1) & (df["concat_correct"] == 0)).sum()
        ),

        "case_all_three_branches_wrong_concat_right": int(
            (
                (df["global_correct"] == 0)
                & (df["part2_correct"] == 0)
                & (df["part4_correct"] == 0)
                & (df["concat_correct"] == 1)
            ).sum()
        ),
        "case_any_branch_right_concat_wrong": int(
            (
                (
                    (df["global_correct"] == 1)
                    | (df["part2_correct"] == 1)
                    | (df["part4_correct"] == 1)
                )
                & (df["concat_correct"] == 0)
            ).sum()
        ),

        "mean_global_conf": float(df["global_conf"].mean()),
        "mean_part2_conf": float(df["part2_conf"].mean()),
        "mean_part4_conf": float(df["part4_conf"].mean()),
        "mean_concat_conf": float(df["concat_conf"].mean()),

        "mean_concat_error_conf": float(concat_error_df["concat_conf"].mean())
        if len(concat_error_df) > 0 else 0.0,
        "median_concat_error_conf": float(concat_error_df["concat_conf"].median())
        if len(concat_error_df) > 0 else 0.0,

        "mean_concat_error_top2_gap": float(concat_error_df["concat_top2_gap"].mean())
        if len(concat_error_df) > 0 else 0.0,
        "median_concat_error_top2_gap": float(concat_error_df["concat_top2_gap"].median())
        if len(concat_error_df) > 0 else 0.0,

        "high_conf_wrong_count_ge_0.9": int((concat_error_df["concat_conf"] >= 0.9).sum()),
        "high_conf_wrong_count_ge_0.8": int((concat_error_df["concat_conf"] >= 0.8).sum()),
        "high_conf_wrong_count_ge_0.7": int((concat_error_df["concat_conf"] >= 0.7).sum()),
    }

    if "mean_selected_part4_score" in df.columns:
        summary["mean_selected_part4_score"] = float(
            df["mean_selected_part4_score"].mean())
        summary["mean_max_selected_part4_score"] = float(
            df["max_selected_part4_score"].mean())

    if fusion_cls_attn_sum is not None and fusion_cls_attn_count > 0:
        mean_cls_attn = fusion_cls_attn_sum / fusion_cls_attn_count

        summary.update({
            "mean_cls_attn_to_cls": float(mean_cls_attn[0].item()),
            "mean_cls_attn_to_global": float(mean_cls_attn[1].item()),
            "mean_cls_attn_to_part2_group_sum": float(mean_cls_attn[2:6].sum().item()),
            "mean_cls_attn_to_part4_group_sum": float(mean_cls_attn[6:].sum().item()),
            "mean_cls_attn_to_part2_group_mean": float(mean_cls_attn[2:6].mean().item()),
            "mean_cls_attn_to_part4_group_mean": float(mean_cls_attn[6:].mean().item()),
            "mean_cls_attn_to_part2_group_max": float(mean_cls_attn[2:6].max().item()),
            "mean_cls_attn_to_part4_group_max": float(mean_cls_attn[6:].max().item()),
        })

        summary.update({
            "dominant_source_global_count": dominant_source_counter["global"],
            "dominant_source_part2_count": dominant_source_counter["part2_group"],
            "dominant_source_part4_count": dominant_source_counter["part4_group"],
        })

    for idx in range(16):
        summary[f"selected_part4_token_{idx}_count"] = int(
            selected_part4_hist[idx].item())

    summary_path = os.path.join(args.save_dir, "analysis_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    per_class_df = build_per_class_stats(df, config["num_classes"])
    per_class_csv_path = os.path.join(
        args.save_dir, "per_class_branch_stats.csv")
    per_class_df.to_csv(per_class_csv_path, index=False)

    print(f"\n===== PMG Analysis ({model_path}) =====")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nSaved files:")
    print(f"- Detailed predictions: {detailed_csv_path}")
    print(f"- Summary json:        {summary_path}")
    print(f"- Per-class stats:     {per_class_csv_path}")


if __name__ == "__main__":
    main()
