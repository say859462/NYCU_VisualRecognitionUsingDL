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


def safe_top2_gap(prob_row: torch.Tensor) -> float:
    if prob_row.numel() < 2:
        return 0.0
    top2_prob, _ = torch.topk(prob_row, k=2, dim=0)
    return float((top2_prob[0] - top2_prob[1]).item())


def build_per_class_stats(df: pd.DataFrame, num_classes: int) -> pd.DataFrame:
    rows = []
    for class_id in range(num_classes):
        class_df = df[df["true_label"] == class_id]
        if len(class_df) == 0:
            rows.append(
                {
                    "class_id": class_id,
                    "num_samples": 0,
                    "global_acc": 0.0,
                    "part2_acc": 0.0,
                    "part4_acc": 0.0,
                    "concat_acc": 0.0,
                }
            )
            continue

        rows.append(
            {
                "class_id": class_id,
                "num_samples": int(len(class_df)),
                "global_acc": float(class_df["global_correct"].mean() * 100.0),
                "part2_acc": float(class_df["part2_correct"].mean() * 100.0),
                "part4_acc": float(class_df["part4_correct"].mean() * 100.0),
                "concat_acc": float(class_df["concat_correct"].mean() * 100.0),
                "mean_concat_conf": float(class_df["concat_conf"].mean()),
                "mean_concat_top2_gap": float(class_df["concat_top2_gap"].mean()),
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Detailed PMG analysis for pairwise interaction fusion")
    parser.add_argument("--config", type=str, default="./config.json")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="./Plot/Analysis_PairwiseFusion")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--resize", type=int, default=576)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as file:
        config = json.load(file)

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = args.model_path if args.model_path is not None else config["best_model_path"]

    val_transform = transforms.Compose([
        transforms.Resize((args.resize, args.resize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_dataset = ImageDataset(root_dir=config["data_dir"], split="val", transform=val_transform)
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
        backbone_name=config.get("backbone_name", "resnet152_partial_res2net"),
    ).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    rows = []
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Analyzing"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model.forward_pmg(images)

            global_prob = torch.softmax(outputs["global_logits"], dim=1)
            part2_prob = torch.softmax(outputs["part2_logits"], dim=1)
            part4_prob = torch.softmax(outputs["part4_logits"], dim=1)
            concat_prob = torch.softmax(outputs["concat_logits"], dim=1)

            global_pred = torch.argmax(global_prob, dim=1)
            part2_pred = torch.argmax(part2_prob, dim=1)
            part4_pred = torch.argmax(part4_prob, dim=1)
            concat_pred = torch.argmax(concat_prob, dim=1)

            for idx in range(labels.size(0)):
                y = labels[idx].item()
                gp = global_pred[idx].item()
                p2 = part2_pred[idx].item()
                p4 = part4_pred[idx].item()
                cp = concat_pred[idx].item()

                rows.append(
                    {
                        "true_label": y,
                        "global_pred": gp,
                        "part2_pred": p2,
                        "part4_pred": p4,
                        "concat_pred": cp,
                        "global_correct": int(gp == y),
                        "part2_correct": int(p2 == y),
                        "part4_correct": int(p4 == y),
                        "concat_correct": int(cp == y),
                        "global_conf": float(global_prob[idx, gp].item()),
                        "part2_conf": float(part2_prob[idx, p2].item()),
                        "part4_conf": float(part4_prob[idx, p4].item()),
                        "concat_conf": float(concat_prob[idx, cp].item()),
                        "concat_top2_gap": safe_top2_gap(concat_prob[idx]),
                        "attention_mean": float(outputs["attention_map"][idx].mean().item()),
                        "attention_max": float(outputs["attention_map"][idx].max().item()),
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.save_dir, "sample_level_analysis.csv"), index=False)

    per_class_df = build_per_class_stats(df, config["num_classes"])
    per_class_df.to_csv(os.path.join(args.save_dir, "per_class_analysis.csv"), index=False)

    global_acc = float(df["global_correct"].mean() * 100.0)
    part2_acc = float(df["part2_correct"].mean() * 100.0)
    part4_acc = float(df["part4_correct"].mean() * 100.0)
    concat_acc = float(df["concat_correct"].mean() * 100.0)

    case_global_wrong_concat_right = int(((df["global_correct"] == 0) & (df["concat_correct"] == 1)).sum())
    case_part2_wrong_concat_right = int(((df["part2_correct"] == 0) & (df["concat_correct"] == 1)).sum())
    case_part4_wrong_concat_right = int(((df["part4_correct"] == 0) & (df["concat_correct"] == 1)).sum())
    case_part2_right_concat_wrong = int(((df["part2_correct"] == 1) & (df["concat_correct"] == 0)).sum())
    case_part4_right_concat_wrong = int(((df["part4_correct"] == 1) & (df["concat_correct"] == 0)).sum())
    high_conf_wrong_count_ge_0_9 = int(((df["concat_correct"] == 0) & (df["concat_conf"] >= 0.9)).sum())
    high_conf_wrong_count_ge_0_8 = int(((df["concat_correct"] == 0) & (df["concat_conf"] >= 0.8)).sum())

    print(f"===== PMG Analysis ({model_path}) =====")
    print(f"backbone_name: {config.get('backbone_name', 'resnet152_partial_res2net')}")
    print(f"resize: {args.resize}")
    print(f"num_samples: {len(df)}")
    print(f"global_acc: {global_acc}")
    print(f"part2_acc: {part2_acc}")
    print(f"part4_acc: {part4_acc}")
    print(f"concat_acc: {concat_acc}")
    print(f"case_global_wrong_concat_right: {case_global_wrong_concat_right}")
    print(f"case_part2_wrong_concat_right: {case_part2_wrong_concat_right}")
    print(f"case_part4_wrong_concat_right: {case_part4_wrong_concat_right}")
    print(f"case_part2_right_concat_wrong: {case_part2_right_concat_wrong}")
    print(f"case_part4_right_concat_wrong: {case_part4_right_concat_wrong}")
    print(f"mean_concat_error_conf: {df[df['concat_correct'] == 0]['concat_conf'].mean() if (df['concat_correct'] == 0).any() else 0.0}")
    print(f"median_concat_error_conf: {df[df['concat_correct'] == 0]['concat_conf'].median() if (df['concat_correct'] == 0).any() else 0.0}")
    print(f"mean_concat_error_top2_gap: {df[df['concat_correct'] == 0]['concat_top2_gap'].mean() if (df['concat_correct'] == 0).any() else 0.0}")
    print(f"high_conf_wrong_count_ge_0.9: {high_conf_wrong_count_ge_0_9}")
    print(f"high_conf_wrong_count_ge_0.8: {high_conf_wrong_count_ge_0_8}")


if __name__ == "__main__":
    main()
