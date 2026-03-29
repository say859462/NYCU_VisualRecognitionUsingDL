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
    x = x.astype(np.float32)
    x = x - x.min()
    x = x / (x.max() + 1e-8)
    return x


def _overlay_heatmap_on_image(
    rgb_img: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45
) -> np.ndarray:
    cmap = plt.get_cmap("jet")
    heatmap_color = cmap(heatmap)[..., :3]
    overlay = (1.0 - alpha) * rgb_img + alpha * heatmap_color
    return np.clip(overlay, 0.0, 1.0)


def _fmt_pred(name, probs, pred_idx):
    return f"{name}:{pred_idx} ({probs[pred_idx].item():.2f})"


def _build_part4_score_grid(part4_token_weights: torch.Tensor) -> np.ndarray:
    score_grid = part4_token_weights.detach().cpu().numpy().reshape(4, 4)
    score_grid = _normalize_map(score_grid)
    return score_grid


def _build_part4_top4_preview_mask(top4_idx: torch.Tensor) -> np.ndarray:
    mask = np.zeros((16,), dtype=np.float32)
    for idx in top4_idx.detach().cpu().tolist():
        mask[idx] = 1.0
    return mask.reshape(4, 4)


def _upsample_grid_to_image(grid_4x4: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    tensor = torch.tensor(grid_4x4, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    up = F.interpolate(
        tensor,
        size=(out_h, out_w),
        mode="nearest"
    )[0, 0].cpu().numpy()
    return up


def _draw_grid_lines(ax, h, w, rows=4, cols=4, color="white", linewidth=0.8, alpha=0.8):
    for r in range(1, rows):
        y = h * r / rows
        ax.axhline(y=y, color=color, linewidth=linewidth, alpha=alpha)
    for c in range(1, cols):
        x = w * c / cols
        ax.axvline(x=x, color=color, linewidth=linewidth, alpha=alpha)


def _annotate_grid_scores(ax, grid_4x4: np.ndarray, color="white", fontsize=9):
    rows, cols = grid_4x4.shape
    for r in range(rows):
        for c in range(cols):
            ax.text(
                c + 0.5,
                r + 0.5,
                f"{grid_4x4[r, c]:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=fontsize,
                fontweight="bold",
            )


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

        if len(activations) == 0:
            raise RuntimeError("No activations captured. Check target_module.")
        if len(gradients) == 0:
            raise RuntimeError("No gradients captured. Check target_module.")

        feat = activations[0]
        grad = gradients[0]

        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = (weights * feat).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
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
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./Plot/Attention_Outputs/ResNet152_PartialRes2Net_SoftSpatialTokenFusion_GridVis",
    )
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
        backbone_name=config.get("backbone_name", "resnet152_partial_res2net"),
    ).to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    eval_resize = config.get("eval_resize", 576)

    preprocess_geo = transforms.Compose([
        transforms.Resize((eval_resize, eval_resize)),
    ])
    preprocess_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])

    for class_id in tqdm(range(config["num_classes"]), desc="Classes Processed"):
        class_dir = os.path.join(args.val_dir, str(class_id))
        if not os.path.exists(class_dir):
            continue

        class_save_dir = os.path.join(args.save_dir, str(class_id))
        os.makedirs(class_save_dir, exist_ok=True)

        all_images = [
            os.path.join(class_dir, file_name)
            for file_name in os.listdir(class_dir)
            if file_name.lower().endswith((".jpg", ".png", ".jpeg"))
        ]
        if not all_images:
            continue

        sampled_image_paths = random.sample(
            all_images,
            min(args.num_samples_per_class, len(all_images)),
        )

        for img_path in sampled_image_paths:
            raw_img = Image.open(img_path).convert("RGB")
            vis_img = preprocess_geo(raw_img)
            input_tensor = preprocess_tensor(vis_img).unsqueeze(0).to(device)

            rgb_img = np.asarray(vis_img).astype(np.float32) / 255.0
            h, w = rgb_img.shape[:2]

            fusion_global_cam, fusion_pred, fusion_probs, outputs = compute_gradcam(
                model=model,
                input_tensor=input_tensor.clone(),
                target_module=model.global_proj,
                target_logits_key="concat_logits",
            )

            fusion_part4_cam, _, _, _ = compute_gradcam(
                model=model,
                input_tensor=input_tensor.clone(),
                target_module=model.part4_proj,
                target_logits_key="concat_logits",
            )

            part4_cam, part4_pred, part4_probs, _ = compute_gradcam(
                model=model,
                input_tensor=input_tensor.clone(),
                target_module=model.part4_proj,
                target_logits_key="part4_logits",
            )

            with torch.no_grad():
                global_probs = torch.softmax(outputs["global_logits"], dim=1)[0].cpu()
                part2_probs = torch.softmax(outputs["part2_logits"], dim=1)[0].cpu()

                global_pred = int(global_probs.argmax().item())
                part2_pred = int(part2_probs.argmax().item())

                fusion_attn = outputs.get("fusion_attn", None)
                part4_token_weights = outputs.get("part4_token_weights", None)
                part4_top4_preview_idx = outputs.get("part4_top4_preview_idx", None)

                cls_attn_text = ""
                preview_text = ""

                if fusion_attn is not None:
                    cls_attn = fusion_attn[0, 0, :].detach().cpu()

                    global_score = float(cls_attn[1].item())
                    part2_mean = float(cls_attn[2:6].mean().item()) if cls_attn[2:6].numel() > 0 else 0.0
                    part4_mean = float(cls_attn[6:].mean().item()) if cls_attn[6:].numel() > 0 else 0.0

                    cls_attn_text = (
                        f" | CLS attn -> "
                        f"CLS:{cls_attn[0]:.2f}, "
                        f"G:{global_score:.2f}, "
                        f"P2mean:{part2_mean:.2f}, "
                        f"P4mean:{part4_mean:.2f}"
                    )

                if part4_top4_preview_idx is not None:
                    preview_text = f" | P4-top4-preview:{part4_top4_preview_idx[0].detach().cpu().tolist()}"

                if part4_token_weights is not None:
                    part4_score_grid = _build_part4_score_grid(part4_token_weights[0])
                else:
                    part4_score_grid = np.zeros((4, 4), dtype=np.float32)

                if part4_top4_preview_idx is not None:
                    part4_preview_mask = _build_part4_top4_preview_mask(part4_top4_preview_idx[0])
                else:
                    part4_preview_mask = np.zeros((4, 4), dtype=np.float32)

            fusion_global_overlay = _overlay_heatmap_on_image(rgb_img, fusion_global_cam, alpha=0.45)
            fusion_part4_overlay = _overlay_heatmap_on_image(rgb_img, fusion_part4_cam, alpha=0.45)
            part4_overlay = _overlay_heatmap_on_image(rgb_img, part4_cam, alpha=0.45)

            part4_score_map_img = _upsample_grid_to_image(part4_score_grid, h, w)
            part4_score_overlay = _overlay_heatmap_on_image(rgb_img, part4_score_map_img, alpha=0.45)

            part4_preview_mask_img = _upsample_grid_to_image(part4_preview_mask, h, w)
            part4_preview_overlay = _overlay_heatmap_on_image(rgb_img, part4_preview_mask_img, alpha=0.40)

            fig, axes = plt.subplots(2, 4, figsize=(24, 12))
            title_color = "green" if str(fusion_pred) == str(class_id) else "red"

            fig.suptitle(
                (
                    f"True: {class_id} | "
                    f"{_fmt_pred('G', global_probs, global_pred)} | "
                    f"{_fmt_pred('P2', part2_probs, part2_pred)} | "
                    f"{_fmt_pred('P4', part4_probs, part4_pred)} | "
                    f"{_fmt_pred('SoftSpatialFusion', fusion_probs, fusion_pred)}"
                    f"{cls_attn_text}"
                    f"{preview_text}"
                ),
                color=title_color,
                fontweight="bold",
                fontsize=13,
            )

            axes[0, 0].imshow(vis_img)
            axes[0, 0].axis("off")
            axes[0, 0].set_title(f"Original Resize{eval_resize}")

            axes[0, 1].imshow(fusion_global_overlay)
            axes[0, 1].axis("off")
            axes[0, 1].set_title("SoftSpatialFusion CAM\n(hook: global_proj)")

            axes[0, 2].imshow(fusion_part4_overlay)
            axes[0, 2].axis("off")
            axes[0, 2].set_title("SoftSpatialFusion CAM\n(hook: part4_proj)")

            axes[0, 3].imshow(part4_overlay)
            axes[0, 3].axis("off")
            axes[0, 3].set_title("Part4 branch CAM\n(hook: part4_proj)")

            axes[1, 0].imshow(part4_score_overlay)
            axes[1, 0].axis("off")
            axes[1, 0].set_title("Part4 token weight heatmap\n(4x4 upsampled overlay)")
            _draw_grid_lines(axes[1, 0], h, w, rows=4, cols=4)

            axes[1, 1].imshow(part4_preview_overlay)
            axes[1, 1].axis("off")
            axes[1, 1].set_title("Top-4 preview mask\n(from soft weights)")
            _draw_grid_lines(axes[1, 1], h, w, rows=4, cols=4)

            im2 = axes[1, 2].imshow(part4_score_grid, cmap="jet", vmin=0.0, vmax=1.0)
            axes[1, 2].set_title("Part4 token weight grid (4x4)")
            axes[1, 2].set_xticks(np.arange(4))
            axes[1, 2].set_yticks(np.arange(4))
            _annotate_grid_scores(axes[1, 2], part4_score_grid, color="white", fontsize=9)
            plt.colorbar(im2, ax=axes[1, 2], fraction=0.046, pad=0.04)

            im3 = axes[1, 3].imshow(part4_preview_mask, cmap="gray", vmin=0.0, vmax=1.0)
            axes[1, 3].set_title("Top-4 preview binary mask (4x4)")
            axes[1, 3].set_xticks(np.arange(4))
            axes[1, 3].set_yticks(np.arange(4))
            _annotate_grid_scores(axes[1, 3], part4_preview_mask, color="red", fontsize=10)
            plt.colorbar(im3, ax=axes[1, 3], fraction=0.046, pad=0.04)

            save_name = (
                f"resnet152_partial_res2net_soft_spatial_token_fusion_gridvis_"
                f"{os.path.splitext(os.path.basename(img_path))[0]}.png"
            )
            plt.tight_layout()
            plt.savefig(os.path.join(class_save_dir, save_name), bbox_inches="tight", dpi=220)
            plt.close(fig)


if __name__ == "__main__":
    main()