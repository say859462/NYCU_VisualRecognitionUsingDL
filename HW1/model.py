import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# --- 核心模組 (RSA, CBP, GeM, NormedLinear) ---

class ResidualSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(ResidualSpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv1(x_cat))
        return x * (1 + attn)

class CompactBilinearPooling(nn.Module):
    def __init__(self, input_dim=512, output_dim=4096):
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
        fft_product = torch.fft.fft(sketch1, dim=1) * torch.fft.fft(sketch2, dim=1)
        cbp = torch.fft.ifft(fft_product, dim=1).real.sum(dim=-1)
        cbp = torch.sign(cbp) * torch.sqrt(torch.abs(cbp) + 1e-5)
        return F.normalize(cbp, p=2, dim=1).to(x.dtype)

class GeM(nn.Module):
    def __init__(self, p=2.5, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * float(p))
        self.eps = eps
    def forward(self, x):
        x_f, p_f = x.float(), self.p.float()
        out = F.avg_pool2d(x_f.clamp(min=self.eps).pow(p_f), (x_f.size(-2), x_f.size(-1))).pow(1.0 / p_f)
        return out.to(x.dtype)

class NormedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
    def forward(self, x):
        return torch.mm(F.normalize(x, p=2, dim=1), F.normalize(self.weight, p=2, dim=0))

# --- 主模型 ---

class ImageClassificationModel(nn.Module):
    def __init__(self, num_classes: int = 100, pretrained: bool = True):
        super(ImageClassificationModel, self).__init__()

        # 1. Backbone (ResNet-152 無修改版本)
        resnet = models.resnet152(weights=models.ResNet152_Weights.DEFAULT if pretrained else None)
        self.backbone_l1_l3 = nn.Sequential(*list(resnet.children())[:7])
        self.backbone_l4 = nn.Sequential(*list(resnet.children())[7:8])

        # 2. 特徵處理
        self.reduce3 = nn.Sequential(nn.Conv2d(1024, 512, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(True))
        self.gem3 = GeM(p=2.5)

        self.rsa = ResidualSpatialAttention(kernel_size=7)
        self.reduce4 = nn.Sequential(nn.Conv2d(2048, 512, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(True))
        self.gem4 = GeM(p=2.5)

        # 3. CBP 提升維度至 4096
        self.output_dim = 4096
        self.cbp = CompactBilinearPooling(input_dim=512, output_dim=self.output_dim)

        # 4. ⭐ 核心改動：增強型 Bottleneck Classifier
        # 透過兩層映射與較高的 Dropout (0.5) 代替 DropPath
        self.embedding_cbp = nn.Sequential(
            nn.Linear(self.output_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.PReLU(),
            nn.Dropout(p=0.5), # 強化 Dropout
            nn.Linear(1024, 512),
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
        p3 = self.gem3(self.reduce3(f3_raw)).flatten(1)

        f4_raw = self.backbone_l4(f3_raw)
        # 空間注意力與主體特徵
        avg_out = torch.mean(f4_raw, dim=1, keepdim=True)
        max_out, _ = torch.max(f4_raw, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        spatial_attn = self.rsa.sigmoid(self.rsa.conv1(x_cat))
        
        f4_att = f4_raw * (1 + spatial_attn)

        f4_reduced = self.reduce4(f4_att)
        p4_gem = self.gem4(f4_reduced).flatten(1)
        p4_cbp = self.cbp(f4_reduced)

        # 分類頭
        embed_cbp = self.embedding_cbp(p4_cbp)
        logits_cbp = self.classifier_cbp(embed_cbp)

        fused_gem = torch.cat([p3, p4_gem], dim=1)
        embed_gem = self.embedding_gem(fused_gem)
        logits_gem = self.classifier_gem(embed_gem)

        if self.training:
            if return_attn: return logits_cbp, logits_gem, torch.mean(f4_att, 1, True)
            return logits_cbp, logits_gem
        else:
            # 推論集成
            logits_ensemble = (logits_cbp * 0.8 + logits_gem * 0.2)
            if return_attn: return logits_ensemble, torch.mean(f4_att, 1, True)
            return logits_ensemble

    def check_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total parameters: {total_params:,}")
        return total_params < 100_000_000