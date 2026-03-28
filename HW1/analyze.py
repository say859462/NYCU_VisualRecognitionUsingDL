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
    correct = sum(int(p == y) for p, y in zip(preds, labels))
    return 100.0 * correct / len(labels)


def safe_top2_gap(prob_row: torch.Tensor) -> float:
    if prob_row.numel() < 2:
        return 0.0
    top2_prob, _ = torch.topk(prob_row, k=2, dim=0)
    return float((top2_prob[0] - top2_prob[1]).item())


def main():
    parser = argparse.ArgumentParser(description="Detailed PMG + router analysis")
    parser.add_argument("--config", type=str, default="./config.json")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="./Plot/Analysis_PurePMG_Router_Resize576")
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
        use_logit_router=config.get("use_logit_router", True),
        router_hidden_dim=config.get("router_hidden_dim", 256),
        router_dropout=config.get("router_dropout", 0.1),
    ).to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    rows = []
    all_labels = []
    global_preds, part2_preds, part4_preds, concat_preds, router_preds = [], [], [], [], []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Analyzing"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model.forward_pmg(images)
            global_logits = outputs["global_logits"]
            part2_logits = outputs["part2_logits"]
            part4_logits = outputs["part4_logits"]
            concat_logits = outputs["concat_logits"]
            router_logits = outputs["router_logits"] if "router_logits" in outputs else concat_logits

            global_prob = torch.softmax(global_logits, dim=1)
            part2_prob = torch.softmax(part2_logits, dim=1)
            part4_prob = torch.softmax(part4_logits, dim=1)
            concat_prob = torch.softmax(concat_logits, dim=1)
            router_prob = torch.softmax(router_logits, dim=1)

            global_pred = torch.argmax(global_prob, dim=1)
            part2_pred = torch.argmax(part2_prob, dim=1)
            part4_pred = torch.argmax(part4_prob, dim=1)
            concat_pred = torch.argmax(concat_prob, dim=1)
            router_pred = torch.argmax(router_prob, dim=1)

            global_preds.extend(global_pred.cpu().tolist())
            part2_preds.extend(part2_pred.cpu().tolist())
            part4_preds.extend(part4_pred.cpu().tolist())
            concat_preds.extend(concat_pred.cpu().tolist())
            router_preds.extend(router_pred.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            router_weights = outputs["router_weights"].cpu() if "router_weights" in outputs else None

            for i in range(labels.size(0)):
                y = labels[i].item()
                gp = global_pred[i].item()
                p2 = part2_pred[i].item()
                p4 = part4_pred[i].item()
                cp = concat_pred[i].item()
                rp = router_pred[i].item()

                rows.append({
                    "true_label": y,
                    "global_pred": gp,
                    "part2_pred": p2,
                    "part4_pred": p4,
                    "concat_pred": cp,
                    "router_pred": rp,
                    "global_correct": int(gp == y),
                    "part2_correct": int(p2 == y),
                    "part4_correct": int(p4 == y),
                    "concat_correct": int(cp == y),
                    "router_correct": int(rp == y),
                    "global_conf": float(global_prob[i, gp].item()),
                    "part2_conf": float(part2_prob[i, p2].item()),
                    "part4_conf": float(part4_prob[i, p4].item()),
                    "concat_conf": float(concat_prob[i, cp].item()),
                    "router_conf": float(router_prob[i, rp].item()),
                    "global_top2_gap": safe_top2_gap(global_prob[i]),
                    "part2_top2_gap": safe_top2_gap(part2_prob[i]),
                    "part4_top2_gap": safe_top2_gap(part4_prob[i]),
                    "concat_top2_gap": safe_top2_gap(concat_prob[i]),
                    "router_top2_gap": safe_top2_gap(router_prob[i]),
                    "router_w_global": float(router_weights[i, 0].item()) if router_weights is not None else 0.0,
                    "router_w_part2": float(router_weights[i, 1].item()) if router_weights is not None else 0.0,
                    "router_w_part4": float(router_weights[i, 2].item()) if router_weights is not None else 0.0,
                    "router_w_concat": float(router_weights[i, 3].item()) if router_weights is not None else 0.0,
                })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.save_dir, "pmg_router_predictions_detailed.csv"), index=False)

    router_error_df = df[df["router_correct"] == 0]
    summary = {
        "resize": args.resize,
        "num_samples": len(all_labels),
        "global_acc": compute_accuracy(global_preds, all_labels),
        "part2_acc": compute_accuracy(part2_preds, all_labels),
        "part4_acc": compute_accuracy(part4_preds, all_labels),
        "concat_acc": compute_accuracy(concat_preds, all_labels),
        "router_acc": compute_accuracy(router_preds, all_labels),
        "case_concat_wrong_router_right": int(((df["concat_correct"] == 0) & (df["router_correct"] == 1)).sum()),
        "case_concat_right_router_wrong": int(((df["concat_correct"] == 1) & (df["router_correct"] == 0)).sum()),
        "case_global_wrong_router_right": int(((df["global_correct"] == 0) & (df["router_correct"] == 1)).sum()),
        "case_part2_wrong_router_right": int(((df["part2_correct"] == 0) & (df["router_correct"] == 1)).sum()),
        "case_part4_wrong_router_right": int(((df["part4_correct"] == 0) & (df["router_correct"] == 1)).sum()),
        "mean_router_error_conf": float(router_error_df["router_conf"].mean()) if len(router_error_df) > 0 else 0.0,
        "median_router_error_conf": float(router_error_df["router_conf"].median()) if len(router_error_df) > 0 else 0.0,
        "mean_router_error_top2_gap": float(router_error_df["router_top2_gap"].mean()) if len(router_error_df) > 0 else 0.0,
        "high_conf_wrong_count_ge_0.9": int((router_error_df["router_conf"] >= 0.9).sum()),
        "high_conf_wrong_count_ge_0.8": int((router_error_df["router_conf"] >= 0.8).sum()),
        "mean_router_w_global": float(df["router_w_global"].mean()) if "router_w_global" in df else 0.0,
        "mean_router_w_part2": float(df["router_w_part2"].mean()) if "router_w_part2" in df else 0.0,
        "mean_router_w_part4": float(df["router_w_part4"].mean()) if "router_w_part4" in df else 0.0,
        "mean_router_w_concat": float(df["router_w_concat"].mean()) if "router_w_concat" in df else 0.0,
    }

    with open(os.path.join(args.save_dir, "analysis_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n===== Pure PMG + Router Analysis Summary ({model_path}) =====")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
