import torch, os, pandas as pd, argparse, json
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from dataset import ImageDataset
from model import ImageClassificationModel

def main():
    parser = argparse.ArgumentParser(description="Final Inference")
    parser.add_argument('--config', type=str, default='./config.json')
    parser.add_argument('--model_path', type=str, default='./Model_Weight/best_model.pth')
    parser.add_argument('--img_size', type=int, default=576)
    parser.add_argument('--tta', type=str, default='none', choices=['none', 'flip', 'rotational'])
    args = parser.parse_args()

    with open(args.config, 'r') as f: config = json.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_transform = transforms.Compose([
        transforms.Resize(576), transforms.CenterCrop(512),
        transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = ImageDataset(root_dir=config['data_dir'], split="test", transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)

    model = ImageClassificationModel(num_classes=config['num_classes'], pretrained=False).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    all_predictions = []
    print(f"🚀 Running Final {args.tta.upper()} TTA Inference...")

    with torch.no_grad():
        for images, _ in tqdm(test_loader, desc="Testing", colour="yellow"):
            images = images.to(device)

            if args.tta == 'none':
                avg_probs = F.softmax(model(images), dim=1)
            elif args.tta == 'flip':
                avg_probs = (F.softmax(model(images), dim=1) + F.softmax(model(torch.flip(images, dims=[3])), dim=1)) / 2.0
            elif args.tta == 'rotational':
                p0 = F.softmax(model(images), dim=1)
                p1 = F.softmax(model(torch.flip(images, dims=[3])), dim=1)
                p2 = F.softmax(model(torch.rot90(images, k=1, dims=[2, 3])), dim=1)
                p3 = F.softmax(model(torch.rot90(images, k=3, dims=[2, 3])), dim=1)
                avg_probs = (p0 + p1 + p2 + p3) / 4.0

            _, preds = torch.max(avg_probs, 1)
            all_predictions.extend(preds.cpu().numpy())

    image_names = [os.path.splitext(os.path.basename(p))[0] for p in test_dataset.image_paths]
    submission_df = pd.DataFrame({'image_name': image_names, 'pred_label': all_predictions})
    submission_df.to_csv("prediction.csv", index=False)
    print(f"\n🎉 Submission CSV saved!")

if __name__ == "__main__":
    main()