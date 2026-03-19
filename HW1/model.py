import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ==============================================================================
# 1. FGVC 增強組件 (完全保留你的獨家設計，並降噪優化)
# ==============================================================================


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
        return x * y


class ResidualSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(ResidualSpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.sigmoid(self.conv1(torch.cat([avg_out, max_out], dim=1)))
        return x * (1 + attn)


class CompactBilinearPooling(nn.Module):
    # 輸出降維至 1024，大幅減少 FFT 帶來的計算雜訊與震盪
    def __init__(self, input_dim=512, output_dim=1024):
        super(CompactBilinearPooling, self).__init__()
        self.output_dim = output_dim
        self.register_buffer('h1', torch.randint(0, output_dim, (input_dim,)))
        self.register_buffer('s1', torch.randint(0, 2, (input_dim,)) * 2 - 1)
        self.register_buffer('h2', torch.randint(0, output_dim, (input_dim,)))
        self.register_buffer('s2', torch.randint(0, 2, (input_dim,)) * 2 - 1)

    def forward(self, x):
        B, C, H, W = x.size()
        x_flat = x.view(B, C, -1).float()
        x1 = x_flat * self.s1.view(1, C, 1).float()
        x2 = x_flat * self.s2.view(1, C, 1).float()
        sketch1 = torch.zeros(B, self.output_dim, H*W, device=x.device).scatter_add_(
            1, self.h1.view(1, C, 1).expand(B, C, H*W), x1)
        sketch2 = torch.zeros(B, self.output_dim, H*W, device=x.device).scatter_add_(
            1, self.h2.view(1, C, 1).expand(B, C, H*W), x2)
        fft_product = torch.fft.fft(
            sketch1, dim=1) * torch.fft.fft(sketch2, dim=1)
        cbp = torch.fft.ifft(fft_product, dim=1).real.sum(dim=-1)
        cbp = torch.sign(cbp) * torch.sqrt(torch.abs(cbp) + 1e-5)
        return F.normalize(cbp, p=2, dim=1).to(x.dtype)


class NormedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        return torch.mm(F.normalize(x, p=2, dim=1), F.normalize(self.weight, p=2, dim=0))


class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p)

# ==============================================================================
# 2. 完整模型 (官方 torchvision 骨幹 + 自訂表頭嫁接)
# ==============================================================================


class ImageClassificationModel(nn.Module):
    def __init__(self, num_classes: int = 100, pretrained: bool = True):
        super(ImageClassificationModel, self).__init__()

        # ---------------------------------------------------------
        # A. 直接提取 torchvision 官方 ResNeXt-101 32x8d
        # ---------------------------------------------------------
        weights = models.ResNeXt50_32X4D_Weights.DEFAULT if pretrained else None
        backbone = models.resnext50_32x4d(weights=weights)

        # 將骨幹拆解並賦值給我們的模組 (丟棄原本的 avgpool 和 fc)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # ---------------------------------------------------------
        # B. FGVC 組件對齊 (Layer 3: SE, Layer 4: RSA)
        # ---------------------------------------------------------
        self.se_l3 = SEBlock(1024)
        self.reduce3 = nn.Sequential(
            nn.Conv2d(1024, 512, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(True))
        self.gem3 = GeM(p=2.5)

        self.rsa_l4 = ResidualSpatialAttention()
        self.reduce4 = nn.Sequential(
            nn.Conv2d(2048, 512, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(True))
        self.gem4 = GeM(p=2.5)

        self.cbp = CompactBilinearPooling(512, 2048)

        # ---------------------------------------------------------
        # C. 決策層
        # ---------------------------------------------------------
        self.embedding_cbp = nn.Sequential(
            nn.Linear(2048, 512), nn.BatchNorm1d(512), nn.PReLU(), nn.Dropout(0.6))

        self.classifier_cbp = NormedLinear(512, num_classes)

        self.embedding_gem = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.PReLU(), nn.Dropout(0.5))
        self.classifier_gem = NormedLinear(512, num_classes)

    def forward(self, x, return_attn=False):
        # 1. 前段基礎特徵流 (直接走官方 layer)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        f3_raw = self.layer3(x)

        # 2. 中階紋理流 (L3): 僅使用通道注意力 (SE)
        f3_att = self.se_l3(f3_raw)
        p3 = self.gem3(self.reduce3(f3_att)).flatten(1)

        # 3. 高階判別流 (L4): 僅使用空間注意力 (RSA) 鎖定主體
        f4_raw = self.layer4(f3_raw)
        f4_att = self.rsa_l4(f4_raw)
        f4_red = self.reduce4(f4_att)

        p4_gem = self.gem4(f4_red).flatten(1)
        p4_cbp = self.cbp(f4_red)

        # 4. 雙表頭融合決策
        embed_cbp = self.embedding_cbp(p4_cbp)
        embed_gem = self.embedding_gem(torch.cat([p3, p4_gem], dim=1))

        logits_cbp = self.classifier_cbp(embed_cbp)
        logits_gem = self.classifier_gem(embed_gem)

        if self.training:
            if return_attn:
                return logits_cbp, logits_gem, embed_cbp, embed_gem, torch.mean(f4_att, dim=1, keepdim=True)
            return logits_cbp, logits_gem, embed_cbp, embed_gem
        else:
            logits_ensemble = (logits_cbp * 0.8 + logits_gem * 0.2)
            if return_attn:
                return logits_ensemble, torch.mean(f4_att, dim=1, keepdim=True)
            return logits_ensemble

    def check_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        print(
            f"📊 Final Model Status: {total/1e6:.2f}M params (Threshold: 100M)")
        return total < 100_000_000
