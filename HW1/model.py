import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class CosineClassifier(nn.Module):
    def __init__(self, in_dim, num_classes):  # 移除了 scale 參數
        super(CosineClassifier, self).__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.weight = nn.Parameter(torch.Tensor(num_classes, in_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        # ⭐ 關鍵：只回傳純粹的 Cosine 相似度 (數值範圍 -1 ~ 1)
        return F.linear(x_norm, w_norm)


class ImageClassificationModel(nn.Module):
    def __init__(self, num_classes: int = 100, pretrained: bool = True):
        super(ImageClassificationModel, self).__init__()

        # 使用 ResNeSt-101 引擎
        backbone = timm.create_model('resnest101e', pretrained=pretrained)

        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.act1, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # PMG 多粒度特徵投影 (⭐ 依建議移除 Dropout，保留純淨特徵)
        self.side2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.PReLU()
        )
        self.side3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.PReLU(),
            nn.Dropout(0.4)
        )
        self.side4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(2048, 512), nn.BatchNorm1d(
                512), nn.PReLU(), nn.Dropout(0.4)
        )

        # ⭐ 最終融合決策層改用 Cosine Classifier
        self.classifier = CosineClassifier(
            in_dim=512 * 3, num_classes=num_classes)

    def extract_features(self, x):
        x = self.stem(x)
        l1 = self.layer1(x)

        l2 = self.layer2(l1)  # 局部細節
        l3 = self.layer3(l2)  # 部位特徵
        l4 = self.layer4(l3)  # 全局輪廓

        f2 = self.side2(l2)
        f3 = self.side3(l3)
        f4 = self.side4(l4)

        return torch.cat([f2, f3, f4], dim=1)

    def forward(self, x):
        emb = self.extract_features(x)
        return self.classifier(emb)

    def check_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        print(f"📊 ResNeSt101-PMG-Cosine Status: {total/1e6:.2f}M params")
        return total < 100_000_000
