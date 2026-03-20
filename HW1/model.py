import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from utils import ResidualSpatialAttention


class GeM(nn.Module):
    def __init__(self, p=2.5, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p)


class NormedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        return torch.mm(F.normalize(x, p=2, dim=1), F.normalize(self.weight, p=2, dim=0))


class ImageClassificationModel(nn.Module):
    def __init__(self, num_classes: int = 100, pretrained: bool = True):
        super(ImageClassificationModel, self).__init__()
        backbone = models.resnext101_32x8d(
            weights=models.ResNeXt101_32X8D_Weights.DEFAULT if pretrained else None)

        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.rsa3 = ResidualSpatialAttention(kernel_size=7)
        self.rsa4 = ResidualSpatialAttention(kernel_size=7)
        self.gem = GeM(p=2.5)

        self.bottleneck = nn.Sequential(
            nn.Linear(3072, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(),
            nn.Dropout(0.5)
        )

        # ⭐ 只需一個分類頭，Global 和 Local 共享權重
        self.classifier = NormedLinear(512, num_classes)

    def extract_features(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)

        f3 = self.layer3(x)
        f3 = self.rsa3(f3)

        f4 = self.layer4(f3)
        f4 = self.rsa4(f4)

        # ⭐ 修正 2：恢復 Layer 3 與 Layer 4 的拼接 (Concatenation)
        pool3 = self.gem(f3).flatten(1)  # [B, 1024]
        pool4 = self.gem(f4).flatten(1)  # [B, 2048]
        fused = torch.cat([pool3, pool4], dim=1)  # [B, 3072]

        emb = self.bottleneck(fused)
        return emb, f4

    def forward(self, x):
        # ⭐ 單純地提取特徵並分類，順便回傳 f4 給 train.py 做裁切
        emb, f4 = self.extract_features(x)
        logits = self.classifier(emb)
        return logits, f4

    def check_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        print(
            f"📊 ResNeXt (Clean Architecture) Status: {total/1e6:.2f}M params")
        return total < 100_000_000
