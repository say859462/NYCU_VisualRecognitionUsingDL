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
from train import _get_stage_weights


def compute_accuracy(preds, labels):
    if len(labels) == 0:
        return 0.0
    correct = sum(int(p == y) for p, y in zip(preds, labels))
    return 100.0 * correct / len(labels)


def main():
    parser = argparse.ArgumentParser(description="PMG / Fusion analysis")
    parser.add_argument("--config", type=str, default="./config.json")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--save_dir", type=str,
                        default="./Plot/Analysis_PMG_74th")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = args.model_path if args.model_path is not None else config["best_model_path"]

    val_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_dataset = ImageDataset(
        root_dir=config["data_dir"],
        split="val",
        transform=val_transform
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model = ImageClassificationModel(
        num_classes=config["num_classes"],
        pretrained=False,
        num_subcenters=config.get("num_subcenters", 3),
        embed_dim=config.get("embed_dim", 256),
        fusion_init_weights=config.get("fusion_init_weights", None),
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    stage_cfg = _get_stage_weights(
        epoch=config.get("num_epochs", 30),
        stage1_epochs=config.get("pmg_stage1_epochs", 4),
        stage2_epochs=config.get("pmg_stage2_epochs", 4),
        config=config,
    )

    rows = []
    all_labels = []

    global_preds = []
    part2_preds = []
    part4_preds = []
    concat_preds = []
    fusion_preds = []

    concat_wrong_fusion_right = 0
    concat_right_fusion_wrong = 0
    fusion_error_conf = []
    fusion_error_top2_gap = []

    fusion_weights_snapshot = None

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Analyzing"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model.forward_pmg(images, stage_cfg=stage_cfg)

            global_logits = outputs["global_logits"]
            part2_logits = outputs["part2_logits"]
            part4_logits = outputs["part4_logits"]
            concat_logits = outputs["concat_logits"]
            fusion_logits = outputs["fusion_logits"]

            if fusion_weights_snapshot is None:
                fusion_weights_snapshot = outputs["fusion_weights"]

            global_prob = torch.softmax(global_logits, dim=1)
            part2_prob = torch.softmax(part2_logits, dim=1)
            part4_prob = torch.softmax(part4_logits, dim=1)
            concat_prob = torch.softmax(concat_logits, dim=1)
            fusion_prob = torch.softmax(fusion_logits, dim=1)

            global_pred = torch.argmax(global_prob, dim=1)
            part2_pred = torch.argmax(part2_prob, dim=1)
            part4_pred = torch.argmax(part4_prob, dim=1)
            concat_pred = torch.argmax(concat_prob, dim=1)
            fusion_pred = torch.argmax(fusion_prob, dim=1)

            global_preds.extend(global_pred.cpu().tolist())
            part2_preds.extend(part2_pred.cpu().tolist())
            part4_preds.extend(part4_pred.cpu().tolist())
            concat_preds.extend(concat_pred.cpu().tolist())
            fusion_preds.extend(fusion_pred.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            top2_prob, top2_idx = torch.topk(fusion_prob, k=2, dim=1)

            for i in range(labels.size(0)):
                y = labels[i].item()
                gp = global_pred[i].item()
                p2 = part2_pred[i].item()
                p4 = part4_pred[i].item()
                cp = concat_pred[i].item()
                fp = fusion_pred[i].item()

                if cp != y and fp == y:
                    concat_wrong_fusion_right += 1
                if cp == y and fp != y:
                    concat_right_fusion_wrong += 1

                if fp != y:
                    fusion_error_conf.append(float(fusion_prob[i, fp].item()))
                    fusion_error_top2_gap.append(
                        float((top2_prob[i, 0] - top2_prob[i, 1]).item()))

                rows.append({
                    "true_label": y,
                    "global_pred": gp,
                    "part2_pred": p2,
                    "part4_pred": p4,
                    "concat_pred": cp,
                    "fusion_pred": fp,
                    "global_conf": float(global_prob[i, gp].item()),
                    "part2_conf": float(part2_prob[i, p2].item()),
                    "part4_conf": float(part4_prob[i, p4].item()),
                    "concat_conf": float(concat_prob[i, cp].item()),
                    "fusion_conf": float(fusion_prob[i, fp].item()),
                })

    pmg_df = pd.DataFrame(rows)
    pmg_df.to_csv(os.path.join(
        args.save_dir, "pmg_predictions.csv"), index=False)

    summary = {
        "num_samples": len(all_labels),
        "global_acc": compute_accuracy(global_preds, all_labels),
        "part2_acc": compute_accuracy(part2_preds, all_labels),
        "part4_acc": compute_accuracy(part4_preds, all_labels),
        "concat_acc": compute_accuracy(concat_preds, all_labels),
        "fusion_acc": compute_accuracy(fusion_preds, all_labels),
        "case_concat_wrong_fusion_right": concat_wrong_fusion_right,
        "case_concat_right_fusion_wrong": concat_right_fusion_wrong,
        "mean_fusion_error_conf": float(sum(fusion_error_conf) / len(fusion_error_conf)) if fusion_error_conf else 0.0,
        "median_fusion_error_conf": float(pd.Series(fusion_error_conf).median()) if fusion_error_conf else 0.0,
        "mean_fusion_error_top2_gap": float(sum(fusion_error_top2_gap) / len(fusion_error_top2_gap)) if fusion_error_top2_gap else 0.0,
        "high_conf_wrong_count_ge_0.9": int(sum(v >= 0.9 for v in fusion_error_conf)),
        "high_conf_wrong_count_ge_0.8": int(sum(v >= 0.8 for v in fusion_error_conf)),
        "fusion_weights": fusion_weights_snapshot,
    }

    summary_path = os.path.join(args.save_dir, "analysis_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n===== Analysis Summary ({model_path}) =====")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
