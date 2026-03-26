import os
import json
import argparse
from collections import defaultdict

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import ImageDataset
from model import ImageClassificationModel


def softmax_probs(logits: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits, dim=1)


def topk_info(probs: torch.Tensor, k: int = 3):
    values, indices = torch.topk(probs, k=k, dim=1)
    return values.cpu().numpy(), indices.cpu().numpy()


@torch.no_grad()
def run_analysis(model, val_loader, device, save_dir: str, hard_pair_topk: int = 20):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    records = []
    confusion_counter = defaultdict(int)

    for batch_idx, batch in enumerate(val_loader):
        images, labels = batch
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model.forward_pmg(images)
        global_logits = outputs["global_logits"]
        part2_logits = outputs["part2_logits"]
        part4_logits = outputs["part4_logits"]
        concat_logits = outputs["concat_logits"]

        global_probs = softmax_probs(global_logits)
        part2_probs = softmax_probs(part2_logits)
        part4_probs = softmax_probs(part4_logits)
        concat_probs = softmax_probs(concat_logits)

        global_topv, global_topi = topk_info(global_probs, k=3)
        part2_topv, part2_topi = topk_info(part2_probs, k=3)
        part4_topv, part4_topi = topk_info(part4_probs, k=3)
        concat_topv, concat_topi = topk_info(concat_probs, k=3)

        global_pred = torch.argmax(global_probs, dim=1)
        part2_pred = torch.argmax(part2_probs, dim=1)
        part4_pred = torch.argmax(part4_probs, dim=1)
        concat_pred = torch.argmax(concat_probs, dim=1)

        batch_size = images.size(0)
        for i in range(batch_size):
            y = labels[i].item()
            p_global = global_pred[i].item()
            p_part2 = part2_pred[i].item()
            p_part4 = part4_pred[i].item()
            p_concat = concat_pred[i].item()

            record = {
                "batch_idx": batch_idx,
                "sample_in_batch": i,
                "true_label": y,
                "global_pred": p_global,
                "global_conf": float(global_probs[i, p_global].item()),
                "part2_pred": p_part2,
                "part2_conf": float(part2_probs[i, p_part2].item()),
                "part4_pred": p_part4,
                "part4_conf": float(part4_probs[i, p_part4].item()),
                "concat_pred": p_concat,
                "concat_conf": float(concat_probs[i, p_concat].item()),
                "concat_top1": int(concat_topi[i, 0]),
                "concat_top1_prob": float(concat_topv[i, 0]),
                "concat_top2": int(concat_topi[i, 1]),
                "concat_top2_prob": float(concat_topv[i, 1]),
                "concat_top3": int(concat_topi[i, 2]),
                "concat_top3_prob": float(concat_topv[i, 2]),
                "global_correct": int(p_global == y),
                "part2_correct": int(p_part2 == y),
                "part4_correct": int(p_part4 == y),
                "concat_correct": int(p_concat == y),
            }
            records.append(record)
            if p_concat != y:
                confusion_counter[(y, p_concat)] += 1

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(save_dir, "pmg_predictions.csv"), index=False)

    summary = {
        "num_samples": len(df),
        "global_acc": float(df["global_correct"].mean() * 100.0),
        "part2_acc": float(df["part2_correct"].mean() * 100.0),
        "part4_acc": float(df["part4_correct"].mean() * 100.0),
        "concat_acc": float(df["concat_correct"].mean() * 100.0),
    }

    cond_global_wrong_concat_right = (
        df["global_correct"] == 0) & (df["concat_correct"] == 1)
    cond_global_right_concat_wrong = (
        df["global_correct"] == 1) & (df["concat_correct"] == 0)
    summary["case_global_wrong_concat_right"] = int(
        cond_global_wrong_concat_right.sum())
    summary["case_global_right_concat_wrong"] = int(
        cond_global_right_concat_wrong.sum())

    concat_error_df = df[df["concat_correct"] == 0].copy()
    if len(concat_error_df) > 0:
        summary["mean_concat_error_conf"] = float(
            concat_error_df["concat_conf"].mean())
        summary["median_concat_error_conf"] = float(
            concat_error_df["concat_conf"].median())
        summary["mean_concat_error_top2_prob"] = float(
            concat_error_df["concat_top2_prob"].mean())
        summary["high_conf_wrong_count_ge_0.9"] = int(
            (concat_error_df["concat_conf"] >= 0.9).sum())
        summary["high_conf_wrong_count_ge_0.8"] = int(
            (concat_error_df["concat_conf"] >= 0.8).sum())
    else:
        summary["mean_concat_error_conf"] = None
        summary["median_concat_error_conf"] = None
        summary["mean_concat_error_top2_prob"] = None
        summary["high_conf_wrong_count_ge_0.9"] = 0
        summary["high_conf_wrong_count_ge_0.8"] = 0

    hard_pairs = sorted(confusion_counter.items(),
                        key=lambda x: x[1], reverse=True)[:hard_pair_topk]
    hard_rows = []
    for (true_cls, pred_cls), count in hard_pairs:
        subset = concat_error_df[(concat_error_df["true_label"] == true_cls) & (
            concat_error_df["concat_pred"] == pred_cls)]
        hard_rows.append({
            "true_class": true_cls,
            "pred_class": pred_cls,
            "error_count": count,
            "mean_conf": float(subset["concat_conf"].mean()) if len(subset) > 0 else None,
            "mean_top2_prob": float(subset["concat_top2_prob"].mean()) if len(subset) > 0 else None,
        })
    pd.DataFrame(hard_rows).to_csv(os.path.join(
        save_dir, "hard_pairs_confidence.csv"), index=False)

    with open(os.path.join(save_dir, "analysis_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Analyze PMG multi-granularity behavior")
    parser.add_argument("--config", type=str, default="./config.json")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--save_dir", type=str,
                        default="./Plot/Analysis_PMG_˙70th")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--hard_pair_topk", type=int, default=20)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = args.model_path if args.model_path is not None else config["best_model_path"]

    val_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    val_dataset = ImageDataset(
        root_dir=config["data_dir"], split="val", transform=val_transform)
    batch_size = args.batch_size if args.batch_size is not None else config["batch_size"]
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    model = ImageClassificationModel(
        num_classes=config["num_classes"],
        pretrained=False,
        num_subcenters=config.get("num_subcenters", 3),
        embed_dim=config.get("embed_dim", 256),
    ).to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    summary = run_analysis(model=model, val_loader=val_loader, device=device,
                           save_dir=args.save_dir, hard_pair_topk=args.hard_pair_topk)
    print(f"\n===== Analysis Summary ({model_path}) =====")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
