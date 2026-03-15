import torch
import os
import pandas as pd
import argparse
import json
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import ImageDataset
from model import ImageClassificationModel


def main():
    parser = argparse.ArgumentParser(
        description="Final Inference for Codebench")
    parser.add_argument('--config', type=str,
                        default='./config.json', help='Path to config')
    parser.add_argument('--model_path', type=str, default='./Model_Weight/16th/best_model.pth',
                        help='Path to your best exp_16 model')
    parser.add_argument('--img_size', type=int, default=512,
                        help='Optimized crop size')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    BATCH_SIZE = config['batch_size']
    NUM_CLASSES = config['num_classes']
    DATA_DIR = config['data_dir']
    OUTPUT_CSV = "prediction.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Device: {device} | TTA: 4-Crop Rotational | Resolution: {args.img_size}")

    # ==========================================
    # 採用實驗證實最穩定的 512 解析度配置
    # ==========================================
    test_transform = transforms.Compose([
        transforms.Resize(int(args.img_size * 1.15)),  # Resize 到約 600
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    test_dataset = ImageDataset(
        root_dir=DATA_DIR, split="test", transform=test_transform)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 載入模型
    model = ImageClassificationModel(
        num_classes=NUM_CLASSES, pretrained=False).to(device)
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"✅ Successfully loaded: {args.model_path}")
    else:
        raise FileNotFoundError(f"Missing weight file at {args.model_path}")

    model.eval()
    all_predictions = []

    print("🚀 Running Final 4-Crop Rotational TTA Inference...")

    with torch.no_grad():
        for images, _ in tqdm(test_loader, desc="Testing", colour="yellow"):
            images = images.to(device)

            # --- 4-Crop Rotational TTA 核心實作 ---
            # 1. 取得不同視角的 Logits
            out_orig = model(images)
            out_flip = model(torch.flip(images, dims=[3]))
            out_rot90 = model(torch.rot90(images, k=1, dims=[2, 3]))
            out_rot270 = model(torch.rot90(images, k=3, dims=[2, 3]))

            # 2. 轉換至機率空間 (Softmax)
            p0 = F.softmax(out_orig, dim=1)
            p1 = F.softmax(out_flip, dim=1)
            p2 = F.softmax(out_rot90, dim=1)
            p3 = F.softmax(out_rot270, dim=1)

            # 3. 平均融合 (Probabilistic Ensemble)
            avg_probs = (p0 + p1 + p2 + p3) / 4.0

            _, preds = torch.max(avg_probs, 1)
            all_predictions.extend(preds.cpu().numpy())

    # 生成預測 CSV
    image_names = [os.path.splitext(os.path.basename(path))[
        0] for path in test_dataset.image_paths]
    submission_df = pd.DataFrame({
        'image_name': image_names,
        'pred_label': all_predictions
    })

    submission_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n🎉 Submission CSV saved: {OUTPUT_CSV}")
    print(submission_df.head())


if __name__ == "__main__":
    main()
