import torch.nn.functional as F
import torch
import torch.nn as nn
from torchvision import models

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class ResidualSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(ResidualSpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv1(x_cat))
        return x * (1 + attn)

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
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class CompactBilinearPooling(nn.Module):
    def __init__(self, input_dim=512, output_dim=2048):
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

        sketch1 = torch.zeros(B, self.output_dim, H * W, dtype=torch.float32, device=x.device)
        sketch1.scatter_add_(1, self.h1.view(1, C, 1).expand(B, C, H * W), x1)

        sketch2 = torch.zeros(B, self.output_dim, H * W, dtype=torch.float32, device=x.device)
        sketch2.scatter_add_(1, self.h2.view(1, C, 1).expand(B, C, H * W), x2)

        fft1 = torch.fft.fft(sketch1, dim=1)
        fft2 = torch.fft.fft(sketch2, dim=1)
        fft_product = fft1 * fft2

        cbp = torch.fft.ifft(fft_product, dim=1).real
        cbp = cbp.sum(dim=-1)
        cbp = torch.sign(cbp) * torch.sqrt(torch.abs(cbp) + 1e-5)
        cbp = F.normalize(cbp, p=2, dim=1)
        return cbp.to(x.dtype)

class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * float(p))
        self.eps = eps

    def forward(self, x):
        x_float = x.float()
        p_float = self.p.float()
        out = F.avg_pool2d(
            x_float.clamp(min=self.eps).pow(p_float),
            (x_float.size(-2), x_float.size(-1))
        ).pow(1.0 / p_float)
        return out.to(x.dtype)

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

        resnet = models.resnet152(weights=models.ResNet152_Weights.DEFAULT if pretrained else None)
        self.backbone_l1_l3 = nn.Sequential(*list(resnet.children())[:7])
        self.backbone_l4 = nn.Sequential(*list(resnet.children())[7:8])

        # --- Layer 3 Processing ---
        self.se_l3 = SEBlock(in_channels=1024, reduction=16)
        self.rsa_l3 = ResidualSpatialAttention(kernel_size=7)
        self.reduce3 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, bias=False),  # 回歸最佳維度 512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.gem3 = GeM(p=3.0)

        # --- Layer 4 Processing ---
        self.se_l4 = SEBlock(in_channels=2048, reduction=16)
        self.rsa_l4 = ResidualSpatialAttention(kernel_size=7)
        self.reduce4 = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1, bias=False),  # 資訊瓶頸過濾雜訊 512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.gem4 = GeM(p=3.0)

        # --- CBP 交互空間 ---
        self.output_dim = 2048  # 回歸最佳穩定維度 2048
        self.cbp = CompactBilinearPooling(input_dim=512, output_dim=self.output_dim)

        # --- 雙表頭決策漏斗 ---
        self.embedding_cbp = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(),
            nn.Dropout(p=0.5)
        )
        self.classifier_cbp = NormedLinear(512, num_classes)

        self.embedding_gem = nn.Sequential(
            nn.Linear(512 + 512, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(),
            nn.Dropout(p=0.5)
        )
        self.classifier_gem = NormedLinear(512, num_classes)

    def forward(self, x, return_attn=False):
        f3_raw = self.backbone_l1_l3(x)
        f3_se = self.se_l3(f3_raw)

        avg_out_l3 = torch.mean(f3_se, dim=1, keepdim=True)
        max_out_l3, _ = torch.max(f3_se, dim=1, keepdim=True)
        x_cat_l3 = torch.cat([avg_out_l3, max_out_l3], dim=1)
        spatial_attn_l3 = self.rsa_l3.sigmoid(self.rsa_l3.conv1(x_cat_l3))
        f3_att = f3_se * (1 + spatial_attn_l3)

        p3 = self.gem3(self.reduce3(f3_att)).flatten(1)  # [B, 512]

        f4_raw = self.backbone_l4(f3_raw)
        f4_se = self.se_l4(f4_raw)

        avg_out_l4 = torch.mean(f4_se, dim=1, keepdim=True)
        max_out_l4, _ = torch.max(f4_se, dim=1, keepdim=True)
        x_cat_l4 = torch.cat([avg_out_l4, max_out_l4], dim=1)
        spatial_attn_l4 = self.rsa_l4.sigmoid(self.rsa_l4.conv1(x_cat_l4))
        f4_att = f4_se * (1 + spatial_attn_l4)

        f4_reduced = self.reduce4(f4_att)  # [B, 512, 14, 14]

        p4_gem = self.gem4(f4_reduced).flatten(1)       # [B, 512]
        p4_cbp = self.cbp(f4_reduced)                   # [B, 2048]

        embed_cbp = self.embedding_cbp(p4_cbp)          # [B, 512]
        logits_cbp = self.classifier_cbp(embed_cbp)

        fused_gem = torch.cat([p3, p4_gem], dim=1)      # [B, 1024]
        embed_gem = self.embedding_gem(fused_gem)       # [B, 512]
        logits_gem = self.classifier_gem(embed_gem)

        if self.training:
            # 訓練階段：回傳 embedding 供 CenterLoss 計算類內距離
            if return_attn:
                activation_map = torch.mean(f4_att, dim=1, keepdim=True)
                return logits_cbp, logits_gem, embed_cbp, embed_gem, activation_map
            return logits_cbp, logits_gem, embed_cbp, embed_gem
        else:
            # 驗證/推論階段：自動 8:2 融合
            logits_ensemble = (logits_cbp * 0.8 + logits_gem * 0.2)
            if return_attn:
                activation_map = torch.mean(f4_att, dim=1, keepdim=True)
                return logits_ensemble, activation_map
            return logits_ensemble

    def check_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total parameters: {total_params:,}")
        if total_params < 100_000_000:
            print("Model size is within the 100M limit.")
            return True
        else:
            print("Warning: Model size exceeds the 100M limit!")
        return False