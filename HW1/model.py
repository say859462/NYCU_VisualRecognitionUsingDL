import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
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

        # 載入強大的多尺度 Res2Net101
        backbone = timm.create_model(
            'res2net101_26w_4s', pretrained=pretrained)

        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.act1, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # 唯一的高階注意力裝甲
        self.rsa = ResidualSpatialAttention(kernel_size=7)
        self.gem = GeM(p=2.5)

        self.bottleneck = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(),
            nn.Dropout(0.4)
        )
        self.classifier = NormedLinear(512, num_classes)

    def extract_features(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        f4 = self.layer4(x)
        f4 = self.rsa(f4)

        pool = self.gem(f4).flatten(1)
        emb = self.bottleneck(pool)
        return emb

    def forward(self, x):
        # ⭐ 移除 f4 回傳，回歸最純粹的 logits 輸出
        emb = self.extract_features(x)
        logits = self.classifier(emb)
        return logits

    def check_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        print(f"📊 Res2Net101 (Pure) Status: {total/1e6:.2f}M params")
        return total < 100_000_000
