import os
import json
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import ImageDataset
from model import ImageClassificationModel
from train import generate_cross_attention_bbox_local_view


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def denormalize_image(img_tensor: torch.Tensor) -> np.ndarray:
    img = img_tensor.detach().cpu() * IMAGENET_STD + IMAGENET_MEAN
    img = torch.clamp(img, 0.0, 1.0)
    return img.permute(1, 2, 0).numpy()


def softmax_probs(logits: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits, dim=1)


def topk_info(probs: torch.Tensor, k: int = 3):
    values, indices = torch.topk(probs, k=k, dim=1)
    return values.cpu().numpy(), indices.cpu().numpy()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


@torch.no_grad()
def run_analysis(
    model,
    val_loader,
    device,
    config,
    save_dir: str,
    max_visualizations: int = 80,
    hard_pair_topk: int = 20,
):
    model.eval()

    ensure_dir(save_dir)
    vis_dir = os.path.join(save_dir, "crop_visualizations")
    ensure_dir(vis_dir)

    records = []
    confusion_counter = defaultdict(int)
    vis_count = 0

    for batch_idx, batch in enumerate(val_loader):
        images, labels = batch
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        local_images = generate_cross_attention_bbox_local_view(
            model=model,
            images=images,
            threshold_ratio=config['local_crop_threshold'],
            padding_ratio=config['local_crop_padding_ratio'],
            min_crop_ratio=config['local_min_crop_ratio'],
            max_crop_ratio=config['local_max_crop_ratio'],
            fallback_crop_ratio=config['local_fallback_crop_ratio']
        )

        full_logits = model(images)
        outputs = model.forward_full_local(images, local_images)

        fused_logits = outputs["fused_logits"]
        local_logits = outputs["local1_logits"]

        full_probs = softmax_probs(full_logits)
        local_probs = softmax_probs(local_logits)
        fused_probs = softmax_probs(fused_logits)

        full_topv, full_topi = topk_info(full_probs, k=3)
        local_topv, local_topi = topk_info(local_probs, k=3)
        fused_topv, fused_topi = topk_info(fused_probs, k=3)

        full_pred = torch.argmax(full_probs, dim=1)
        local_pred = torch.argmax(local_probs, dim=1)
        fused_pred = torch.argmax(fused_probs, dim=1)

        batch_size = images.size(0)
        for i in range(batch_size):
            y = labels[i].item()
            p_full = full_pred[i].item()
            p_local = local_pred[i].item()
            p_fused = fused_pred[i].item()

            record = {
                "batch_idx": batch_idx,
                "sample_in_batch": i,
                "true_label": y,

                "full_pred": p_full,
                "full_conf": float(full_probs[i, p_full].item()),
                "full_top1": int(full_topi[i, 0]),
                "full_top1_prob": float(full_topv[i, 0]),
                "full_top2": int(full_topi[i, 1]),
                "full_top2_prob": float(full_topv[i, 1]),
                "full_top3": int(full_topi[i, 2]),
                "full_top3_prob": float(full_topv[i, 2]),

                "local_pred": p_local,
                "local_conf": float(local_probs[i, p_local].item()),
                "local_top1": int(local_topi[i, 0]),
                "local_top1_prob": float(local_topv[i, 0]),
                "local_top2": int(local_topi[i, 1]),
                "local_top2_prob": float(local_topv[i, 1]),
                "local_top3": int(local_topi[i, 2]),
                "local_top3_prob": float(local_topv[i, 2]),

                "fused_pred": p_fused,
                "fused_conf": float(fused_probs[i, p_fused].item()),
                "fused_top1": int(fused_topi[i, 0]),
                "fused_top1_prob": float(fused_topv[i, 0]),
                "fused_top2": int(fused_topi[i, 1]),
                "fused_top2_prob": float(fused_topv[i, 1]),
                "fused_top3": int(fused_topi[i, 2]),
                "fused_top3_prob": float(fused_topv[i, 2]),

                "full_correct": int(p_full == y),
                "local_correct": int(p_local == y),
                "fused_correct": int(p_fused == y),
            }
            records.append(record)

            if p_fused != y:
                confusion_counter[(y, p_fused)] += 1

            if vis_count < max_visualizations:
                img_np = denormalize_image(images[i])
                local_np = denormalize_image(local_images[i])

                fig, axes = plt.subplots(1, 2, figsize=(8, 4))
                axes[0].imshow(img_np)
                axes[0].set_title(
                    f"Full\nT:{y} | F:{p_full}({full_probs[i, p_full]:.2f})"
                )
                axes[0].axis("off")

                axes[1].imshow(local_np)
                axes[1].set_title(
                    f"Local\nL:{p_local}({local_probs[i, p_local]:.2f}) | "
                    f"Fu:{p_fused}({fused_probs[i, p_fused]:.2f})"
                )
                axes[1].axis("off")

                plt.tight_layout()
                plt.savefig(
                    os.path.join(
                        vis_dir, f"sample_{vis_count:04d}_true_{y}.png"),
                    dpi=200
                )
                plt.close(fig)
                vis_count += 1

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(
        save_dir, "full_local_fused_predictions.csv"), index=False)

    summary = {}
    summary["num_samples"] = len(df)
    summary["full_acc"] = float(df["full_correct"].mean() * 100.0)
    summary["local_acc"] = float(df["local_correct"].mean() * 100.0)
    summary["fused_acc"] = float(df["fused_correct"].mean() * 100.0)

    cond_full_wrong_local_right_fused_right = (
        (df["full_correct"] == 0) &
        (df["local_correct"] == 1) &
        (df["fused_correct"] == 1)
    )
    cond_full_wrong_local_wrong_fused_wrong = (
        (df["full_correct"] == 0) &
        (df["local_correct"] == 0) &
        (df["fused_correct"] == 0)
    )
    cond_full_wrong_local_right_fused_wrong = (
        (df["full_correct"] == 0) &
        (df["local_correct"] == 1) &
        (df["fused_correct"] == 0)
    )
    cond_full_right_local_wrong_fused_right = (
        (df["full_correct"] == 1) &
        (df["local_correct"] == 0) &
        (df["fused_correct"] == 1)
    )

    summary["case_full_wrong_local_right_fused_right"] = int(
        cond_full_wrong_local_right_fused_right.sum()
    )
    summary["case_full_wrong_local_wrong_fused_wrong"] = int(
        cond_full_wrong_local_wrong_fused_wrong.sum()
    )
    summary["case_full_wrong_local_right_fused_wrong"] = int(
        cond_full_wrong_local_right_fused_wrong.sum()
    )
    summary["case_full_right_local_wrong_fused_right"] = int(
        cond_full_right_local_wrong_fused_right.sum()
    )

    fused_error_df = df[df["fused_correct"] == 0].copy()
    if len(fused_error_df) > 0:
        summary["mean_fused_error_conf"] = float(
            fused_error_df["fused_conf"].mean())
        summary["median_fused_error_conf"] = float(
            fused_error_df["fused_conf"].median())
        summary["mean_fused_error_top2_prob"] = float(
            fused_error_df["fused_top2_prob"].mean())
        summary["high_conf_wrong_count_ge_0.9"] = int(
            (fused_error_df["fused_conf"] >= 0.9).sum())
        summary["high_conf_wrong_count_ge_0.8"] = int(
            (fused_error_df["fused_conf"] >= 0.8).sum())
    else:
        summary["mean_fused_error_conf"] = None
        summary["median_fused_error_conf"] = None
        summary["mean_fused_error_top2_prob"] = None
        summary["high_conf_wrong_count_ge_0.9"] = 0
        summary["high_conf_wrong_count_ge_0.8"] = 0

    hard_pairs = sorted(
        confusion_counter.items(),
        key=lambda x: x[1],
        reverse=True
    )[:hard_pair_topk]

    hard_pair_rows = []
    for (true_cls, pred_cls), count in hard_pairs:
        subset = fused_error_df[
            (fused_error_df["true_label"] == true_cls) &
            (fused_error_df["fused_pred"] == pred_cls)
        ]
        hard_pair_rows.append({
            "true_class": true_cls,
            "pred_class": pred_cls,
            "error_count": count,
            "mean_conf": float(subset["fused_conf"].mean()) if len(subset) > 0 else None,
            "mean_top2_prob": float(subset["fused_top2_prob"].mean()) if len(subset) > 0 else None,
        })

    hard_df = pd.DataFrame(hard_pair_rows)
    hard_df.to_csv(os.path.join(
        save_dir, "hard_pairs_confidence.csv"), index=False)

    with open(os.path.join(save_dir, "analysis_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    df[cond_full_wrong_local_right_fused_right].to_csv(
        os.path.join(save_dir, "cases_full_wrong_local_right_fused_right.csv"),
        index=False
    )
    df[cond_full_wrong_local_right_fused_wrong].to_csv(
        os.path.join(save_dir, "cases_full_wrong_local_right_fused_wrong.csv"),
        index=False
    )
    df[cond_full_wrong_local_wrong_fused_wrong].to_csv(
        os.path.join(save_dir, "cases_full_wrong_local_wrong_fused_wrong.csv"),
        index=False
    )

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Analyze full / local / fused behavior under staged training"
    )
    parser.add_argument("--config", type=str, default="./config.json")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--save_dir", type=str,
                        default="./Plot/Analysis_StagedLocalCrop_65th")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_visualizations", type=int, default=80)
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
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_dataset = ImageDataset(
        root_dir=config["data_dir"],
        split="val",
        transform=val_transform
    )

    batch_size = args.batch_size if args.batch_size is not None else config["batch_size"]

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model = ImageClassificationModel(
        num_classes=config["num_classes"],
        pretrained=False,
        num_subcenters=config.get("num_subcenters", 3),
        embed_dim=config.get("embed_dim", 256)
    ).to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    summary = run_analysis(
        model=model,
        val_loader=val_loader,
        device=device,
        config=config,
        save_dir=args.save_dir,
        max_visualizations=args.max_visualizations,
        hard_pair_topk=args.hard_pair_topk,
    )

    print(f"\n===== Analysis Summary ({model_path}) =====")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
