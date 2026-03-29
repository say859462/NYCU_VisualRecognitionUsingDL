import argparse
import json
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import ImageDataset
from model import ImageClassificationModel


def main():
    parser = argparse.ArgumentParser(
        description="Final Inference for ResNet152 + partial Res2Net + spatial-token fusion"
    )
    parser.add_argument("--config", type=str, default="./config.json")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--output_csv", type=str, default="prediction.csv")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    model_path = (
        args.model_path
        if args.model_path is not None
        else config["best_model_path"]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eval_resize = config.get("eval_resize", 576)

    test_transform = transforms.Compose([
        transforms.Resize((eval_resize, eval_resize)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    test_dataset = ImageDataset(
        root_dir=config["data_dir"],
        split="test",
        transform=test_transform,
    )
    test_loader = DataLoader(
        test_dataset,
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
    ).to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    all_predictions = []

    print(f"Running final inference from: {model_path}")
    print(f"Test resize: {eval_resize}")

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device, non_blocking=True)

            outputs = model.forward_pmg(images)
            final_logits = outputs["concat_logits"]
            preds = torch.argmax(final_logits, dim=1)

            all_predictions.extend(preds.cpu().tolist())

    image_names = [
        os.path.splitext(os.path.basename(path))[0]
        for path in test_dataset.image_paths
    ]

    submission_df = pd.DataFrame({
        "image_name": image_names,
        "pred_label": all_predictions,
    })
    submission_df.to_csv(args.output_csv, index=False)

    print(f"Submission CSV saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
