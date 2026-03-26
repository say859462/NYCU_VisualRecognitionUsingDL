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
    parser = argparse.ArgumentParser(description="Final Inference")
    parser.add_argument('--config', type=str, default='./config.json')
    parser.add_argument('--model_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    model_path = args.model_path if args.model_path is not None else config['best_model_path']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = ImageDataset(
        root_dir=config['data_dir'], split="test", transform=test_transform)
    test_loader = DataLoader(
        test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

    model = ImageClassificationModel(
        num_classes=config['num_classes'],
        pretrained=False,
        num_subcenters=config.get('num_subcenters', 3),
        embed_dim=config.get('embed_dim', 256)
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_predictions = []
    print(f"🚀 Running Final Inference from: {model_path}")

    import cv2
    import torch.nn.functional as F

    with torch.no_grad():
        outputs = model.forward_pmg(input_tensor)
        global_probs = torch.softmax(outputs['global_logits'], dim=1)[0]
        part2_probs = torch.softmax(outputs['part2_logits'], dim=1)[0]
        part4_probs = torch.softmax(outputs['part4_logits'], dim=1)[0]
        concat_probs = torch.softmax(outputs['concat_logits'], dim=1)[0]

        saliency = model.get_saliency(input_tensor)  # [B,H,W] or [B,1,H,W]
        if saliency.dim() == 4:
            saliency = saliency[:, 0]
        saliency = saliency[0].cpu().numpy()

    saliency = (saliency - saliency.min()) / \
        (saliency.max() - saliency.min() + 1e-8)
    saliency = cv2.resize(saliency, (448, 448), interpolation=cv2.INTER_CUBIC)
    saliency = cv2.GaussianBlur(saliency, (0, 0), sigmaX=8, sigmaY=8)
    saliency = (saliency - saliency.min()) / \
        (saliency.max() - saliency.min() + 1e-8)

    image_names = [os.path.splitext(os.path.basename(p))[0]
                   for p in test_dataset.image_paths]
    submission_df = pd.DataFrame(
        {'image_name': image_names, 'pred_label': all_predictions})
    submission_df.to_csv("prediction.csv", index=False)
    print("\n🎉 Submission CSV saved!")


if __name__ == "__main__":
    main()
