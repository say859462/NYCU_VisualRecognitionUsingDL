import torch
import numpy as np
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms

# 載入您的自定義模組
from dataset import ImageDataset
from model import ImageClassificationModel


def search_best_ensemble_weights(model, dataloader, device):
    """
    使用 PyTorch Hooks 安全攔截 CBP 與 GeM 的原始輸出，
    並掃描最佳的融合比例。
    """
    model.eval()

    # ==========================================
    # 1. 建立 Hooks 攔截系統
    # ==========================================
    activations = {}

    def get_activation(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook

    # 將鉤子掛在兩個分類器上
    handle_cbp = model.classifier_cbp.register_forward_hook(
        get_activation('cbp'))
    handle_gem = model.classifier_gem.register_forward_hook(
        get_activation('gem'))

    all_logits_cbp = []
    all_logits_gem = []
    all_labels = []

    print("Extracting pure Logits using Hooks...")
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Inference"):
            images = images.to(device)

            # 2. 執行推論 (我們不需要接住 return 的融合結果，Hooks 會自動抓取)
            _ = model(images)

            # 3. 收集攔截到的原始 Logits
            all_logits_cbp.append(activations['cbp'].cpu().numpy())
            all_logits_gem.append(activations['gem'].cpu().numpy())
            all_labels.append(labels.numpy())

    # 拔除鉤子，釋放記憶體
    handle_cbp.remove()
    handle_gem.remove()

    # ==========================================
    # 4. 開始網格搜索 (Grid Search)
    # ==========================================
    all_logits_cbp = np.concatenate(all_logits_cbp, axis=0)
    all_logits_gem = np.concatenate(all_logits_gem, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    print("\n--- Starting Grid Search for Ensemble Weights ---")
    best_acc = 0.0
    best_weight_cbp = 0.0

    # 從 0.0 到 1.0，每次遞增 0.05 進行掃描
    for cbp_weight in np.arange(0.0, 1.05, 0.05):
        gem_weight = 1.0 - cbp_weight

        # 融合 Logits
        fused_logits = (cbp_weight * all_logits_cbp) + \
            (gem_weight * all_logits_gem)
        preds = np.argmax(fused_logits, axis=1)

        acc = (np.sum(preds == all_labels) / len(all_labels)) * 100
        print(
            f"CBP Weight: {cbp_weight:.2f} | GeM Weight: {gem_weight:.2f} => Acc: {acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            best_weight_cbp = cbp_weight

    # ==========================================
    # 5. 輸出最終結果
    # ==========================================
    print("\n" + "="*50)
    print("🏆 Grid Search Completed!")
    print(f"Best Configuration:")
    print(f"CBP Weight (紋理分支): {best_weight_cbp:.2f}")
    print(f"GeM Weight (定位分支): {1.0 - best_weight_cbp:.2f}")
    print(f"Maximum Validation Accuracy: {best_acc:.2f}%")
    print("="*50)


def main():
    parser = argparse.ArgumentParser(description="Ensemble Weight Grid Search")
    parser.add_argument('--model_path', type=str,
                        default='./Model_Weight/best_model.pth')
    parser.add_argument('--data_dir', type=str, default='./Dataset/data')
    parser.add_argument('--img_size', type=int, default=576)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 對齊 EXP 30 的 576D 終極解析度
    val_transform = transforms.Compose([
        transforms.Resize(640),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    val_dataset = ImageDataset(
        root_dir=args.data_dir, split="val", transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=16,
                            shuffle=False, num_workers=4)

    model = ImageClassificationModel(
        num_classes=100, pretrained=False).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"Loaded weights from {args.model_path}")

    search_best_ensemble_weights(model, val_loader, device)


if __name__ == "__main__":
    main()
