import torch
import os
import pandas as pd
import argparse
import json
from utils import ProcessCrops
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import ImageDataset
from model import ImageClassificationModel


def main():

    parser = argparse.ArgumentParser(
        description="Image Classification Model Testing")
    parser.add_argument('--config', type=str, default='./config.json',
                        help='Path to the JSON configuration file')
    args = parser.parse_args()

    print(f"Loading configuration file....: {args.config}")
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Training Parameters
    BATCH_SIZE = config['batch_size']
    NUM_CLASSES = config['num_classes']
    DATA_DIR = config['data_dir']
    BEST_MODEL_PATH = './Model_Weight/best_model.pth'
    OUTPUT_CSV = "prediction.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Test Dataset and DataLoader

    test_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    test_dataset = ImageDataset(
        root_dir=DATA_DIR, split="test", transform=test_transform)

    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Model
    model = ImageClassificationModel(
        num_classes=NUM_CLASSES, pretrained=False).to(device)

    if os.path.exists(BEST_MODEL_PATH):
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
        print(f"Loading best model weight from: {BEST_MODEL_PATH}")
    else:
        raise FileNotFoundError(f"Couldn't find {BEST_MODEL_PATH}")

    # Inference
    model.eval()
    all_predictions = []

    print("Predicting on test dataset...")

    with torch.no_grad():
        for images, _ in tqdm(test_loader, desc="Predicting"):
            # images shape: [Batch_Size, 10, 3, H, W]
            images = images.to(device)
            # Multi-crop inference

            outputs = model(images)

            _, preds = torch.max(outputs, 1)

            all_predictions.extend(preds.cpu().numpy())

    # Generate submission CSV
    # Extracting image names from the dataset's image paths
    image_names = [os.path.splitext(os.path.basename(path))[
        0] for path in test_dataset.image_paths]

    submission_df = pd.DataFrame({
        'image_name': image_names,
        'pred_label': all_predictions
    })

    submission_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n Submission saved to: {OUTPUT_CSV}")

    print("\nPrediction Results Preview (First 5 Entries):")
    print(submission_df.head())


if __name__ == "__main__":
    main()
