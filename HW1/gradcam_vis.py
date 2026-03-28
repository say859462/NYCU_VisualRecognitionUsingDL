import argparse
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from model import ImageClassificationModel


def _normalize_map(x: np.ndarray) -> np.ndarray:
    x = x - x.min()
    x = x / (x.max() + 1e-8)
    return x


def _overlay_heatmap_on_image(rgb_img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    cmap = plt.get_cmap("jet")
    heatmap_color = cmap(heatmap)[..., :3]
    overlay = (1 - alpha) * rgb_img + alpha * heatmap_color
    overlay = np.clip(overlay, 0.0, 1.0)
    return overlay


def _fmt_pred(name, probs, pred_idx):
    return f"{name}:{pred_idx} ({probs[pred_idx].item():.2f})"


def compute_gradcam(model, input_tensor, target_module, target_logits_key):
    model.eval()
    activations = []
    gradients = []

    def forward_hook(module, inp, out):
        activations.append(out)

    def backward_hook(module, grad_input, grad_output):
        if grad_output is not None and len(grad_output) > 0 and grad_output[0] is not None:
            gradients.append(grad_output[0])

    handle_fwd = target_module.register_forward_hook(forward_hook)
    handle_bwd = target_module.register_full_backward_hook(backward_hook)

    try:
        input_tensor = input_tensor.requires_grad_(True)
        outputs = model.forward_pmg(input_tensor)
        logits = outputs[target_logits_key]
        pred_idx = logits.argmax(dim=1).item()
        score = logits[:, pred_idx].sum()
        model.zero_grad(set_to_none=True)
        score.backward()

        feat = activations[0]
        grad = gradients[0]
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = (weights * feat).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        cam = _normalize_map(cam)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu()
        return cam, pred_idx, probs, outputs
    finally:
        handle_fwd.remove()
        handle_bwd.remove()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config.json")
    parser.add_argument("--val_dir", type=str, default="./Dataset/data/val")
    parser.add_argument("--num_samples_per_class", type=int, default=3)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--save_dir", type=str,
                        default="./Plot/Attention_Outputs/PurePMG_Res2Net50")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    model_path = args.model_path if args.model_path is not None else config["best_model_path"]

    model = ImageClassificationModel(
        num_classes=config["num_classes"],
        pretrained=False,
        num_subcenters=config.get("num_subcenters", 3),
        embed_dim=config.get("embed_dim", 256),
        use_logit_router=config.get("use_logit_router", False),
        router_hidden_dim=config.get("router_hidden_dim", 256),
        router_dropout=config.get("router_dropout", 0.1),
        backbone_name=config.get("backbone_name", "res2net50_26w_4s"),
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    preprocess_geo = transforms.Compose([
        transforms.Resize((576, 576))
    ])
    preprocess_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    for class_id in tqdm(range(config["num_classes"]), desc="Classes Processed"):
        class_dir = os.path.join(args.val_dir, str(class_id))
        if not os.path.exists(class_dir):
            continue

        class_save_dir = os.path.join(args.save_dir, str(class_id))
        os.makedirs(class_save_dir, exist_ok=True)

        all_images = [
            os.path.join(class_dir, f)
            for f in os.listdir(class_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]
        if not all_images:
            continue

        sampled_image_paths = random.sample(
            all_images, min(args.num_samples_per_class, len(all_images))
        )

        for img_path in sampled_image_paths:
            raw_img = Image.open(img_path).convert("RGB")
            vis_img = preprocess_geo(raw_img)
            input_tensor = preprocess_tensor(vis_img).unsqueeze(0).to(device)
            rgb_img = np.asarray(vis_img).astype(np.float32) / 255.0

            concat_cam, concat_pred, concat_probs, outputs = compute_gradcam(
                model=model,
                input_tensor=input_tensor.clone(),
                target_module=model.fuse,
                target_logits_key="concat_logits",
            )

            part4_cam, part4_pred, part4_probs, _ = compute_gradcam(
                model=model,
                input_tensor=input_tensor.clone(),
                target_module=model.fuse,
                target_logits_key="part4_logits",
            )

            with torch.no_grad():
                global_probs = torch.softmax(
                    outputs["global_logits"], dim=1)[0].cpu()
                part2_probs = torch.softmax(
                    outputs["part2_logits"], dim=1)[0].cpu()
                global_pred = int(global_probs.argmax().item())
                part2_pred = int(part2_probs.argmax().item())

            concat_overlay = _overlay_heatmap_on_image(
                rgb_img, concat_cam, alpha=0.45)
            part4_overlay = _overlay_heatmap_on_image(
                rgb_img, part4_cam, alpha=0.45)

            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            title_color = "green" if str(
                concat_pred) == str(class_id) else "red"

            fig.suptitle(
                (
                    f"True: {class_id} | "
                    f"{_fmt_pred('G', global_probs, global_pred)} | "
                    f"{_fmt_pred('P2', part2_probs, part2_pred)} | "
                    f"{_fmt_pred('P4', part4_probs, part4_pred)} | "
                    f"{_fmt_pred('Concat', concat_probs, concat_pred)}"
                ),
                color=title_color,
                fontweight="bold",
                fontsize=12,
            )

            axes[0].imshow(vis_img)
            axes[0].axis("off")
            axes[0].set_title("Original Resize576")

            axes[1].imshow(concat_overlay)
            axes[1].axis("off")
            axes[1].set_title("Concat Grad-CAM")

            axes[2].imshow(part4_overlay)
            axes[2].axis("off")
            axes[2].set_title("Part4 Grad-CAM")

            save_name = f"pure_pmg_res2net50_{os.path.splitext(os.path.basename(img_path))[0]}.png"
            plt.tight_layout()
            plt.savefig(os.path.join(class_save_dir, save_name),
                        bbox_inches="tight")
            plt.close(fig)


if __name__ == "__main__":
    main()
