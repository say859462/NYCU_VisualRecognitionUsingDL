import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p)

class ImageClassificationModel(nn.Module):
    def __init__(self, num_classes: int = 100, pretrained: bool = True):
        super(ImageClassificationModel, self).__init__()

        weights = models.ResNeXt101_32X8D_Weights.DEFAULT if pretrained else None
        backbone = models.resnext101_32x8d(weights=weights)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # ⭐ 多尺度特徵融合：為 Layer 3 和 Layer 4 分別準備 GeM 池化
        self.gem3 = GeM(p=3.0)
        self.gem4 = GeM(p=3.0)

        # Layer 3 輸出 1024 維，Layer 4 輸出 2048 維，拼接後為 3072 維
        # 透過資訊漏斗壓縮至 512 維，保留精華細節並防範過擬合
        self.embedding = nn.Sequential(
            nn.Linear(1024 + 2048, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(),
            nn.Dropout(0.4)
        )
        
        # 換回最純粹的標準分類器
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x, return_attn=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        
        # 提取 Layer 3 (細緻紋理) 與 Layer 4 (高階語義)
        f3 = self.layer3(x)      # [B, 1024, H', W']
        f4 = self.layer4(f3)     # [B, 2048, H'', W'']

        # 雙層池化與拼接
        p3 = self.gem3(f3).flatten(1)
        p4 = self.gem4(f4).flatten(1)
        p_cat = torch.cat([p3, p4], dim=1)  # [B, 3072]

        embed = self.embedding(p_cat)
        logits = self.classifier(embed)

        if self.training:
            return logits, embed
        else:
            return logits

    def check_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        print(f"📊 Pure ResNeXt (Multi-Scale) Status: {total/1e6:.2f}M params")
        return total < 100_000_000