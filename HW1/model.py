import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math


class CosineLinear(nn.Module):
    """Cosine Classifier: 將內積空間轉換為餘弦空間，配合度量學習 Loss 使用"""

    def __init__(self, in_features, out_features):
        super(CosineLinear, self).__init__()
        self.weight = nn.Parameter(
            torch.FloatTensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        return F.linear(x_norm, w_norm)


class AttentionPooling(nn.Module):
    """線性特徵聚合：保留空間重要性，輸出高斯分佈特徵，完美契合超球面"""

    def __init__(self, in_dim=512):
        super(AttentionPooling, self).__init__()
        self.attn_conv = nn.Conv2d(in_dim, 1, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        attn = self.attn_conv(x)  # [B, 1, H, W]
        attn = F.softmax(attn.view(B, -1), dim=1).view(B, 1, H, W)
        return (x * attn).sum(dim=[2, 3])  # [B, C]


class SimpleTransformer(nn.Module):
    """Transformer Block：支援 2D Positional Encoding，保留完整 512D 空間"""

    def __init__(self, dim=512, num_heads=8, feat_size=14):
        super(SimpleTransformer, self).__init__()
        # 加入可學習的絕對位置編碼 (假設輸入圖片 448x448，Layer4 輸出通常為 14x14)
        self.pos_embed = nn.Parameter(torch.randn(
            1, feat_size * feat_size, dim) * 0.02)

        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 4, dim)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).permute(0, 2, 1)  # [B, H*W, C]

        # 動態適應不同解析度 (若推理時輸入大小改變，進行雙線性插值)
        if x_flat.shape[1] != self.pos_embed.shape[1]:
            pos_embed_rescaled = F.interpolate(
                self.pos_embed.permute(0, 2, 1).view(
                    1, C, int(math.sqrt(self.pos_embed.shape[1])), -1),
                size=(H, W), mode='bilinear', align_corners=False
            ).flatten(2).permute(0, 2, 1)
            x_flat = x_flat + pos_embed_rescaled
        else:
            x_flat = x_flat + self.pos_embed

        attn_out, _ = self.attn(x_flat, x_flat, x_flat)
        x_flat = self.norm1(x_flat + attn_out)

        ffn_out = self.ffn(x_flat)
        x_flat = self.norm2(x_flat + ffn_out)

        x_out = x_flat.permute(0, 2, 1).view(B, C, H, W)
        return x + x_out  # Residual connection

    # 替換 model.py 內的這兩個函數
    def freeze_features_for_crt(self):
        """凍結 Backbone 與 Transformer，但保留 Embedding 層與分類頭解凍"""
        for name, param in self.named_parameters():
            # 只要不是 embedding 層 (emb_l3, emb_l4, emb_fused) 或分類頭 (cls_)，就凍結
            if not any(x in name for x in ['emb_', 'cls_']):
                param.requires_grad = False
            else:
                param.requires_grad = True

    def get_classifier_parameters(self):
        # 讓優化器同時接管 Embedding 層與分類頭
        return (list(self.emb_l3.parameters()) + list(self.cls_l3.parameters()) +
                list(self.emb_l4.parameters()) + list(self.cls_l4.parameters()) +
                list(self.emb_fused.parameters()) + list(self.cls_fused.parameters()))


class SKConv(nn.Module):
    def __init__(self, features, reduction=16):
        super(SKConv, self).__init__()
        self.conv1 = nn.Conv2d(features, features, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(features, features, 3,
                               padding=2, dilation=2, bias=False)
        self.bn1 = nn.BatchNorm2d(features)
        self.bn2 = nn.BatchNorm2d(features)
        self.relu = nn.ReLU(inplace=True)
        d = max(int(features / reduction), 32)
        self.fc1 = nn.Conv2d(features, d, 1, bias=False)
        self.bn_fc1 = nn.BatchNorm2d(d)
        self.fc2 = nn.Conv2d(d, features * 2, 1, bias=False)

    def forward(self, x):
        U1 = self.relu(self.bn1(self.conv1(x)))
        U2 = self.relu(self.bn2(self.conv2(x)))
        U = U1 + U2
        S = U.mean([-2, -1], keepdim=True)
        Z = self.relu(self.bn_fc1(self.fc1(S)))
        A = self.fc2(Z).view(x.size(0), 2, x.size(1), 1, 1)
        A = F.softmax(A, dim=1)
        return A[:, 0] * U1 + A[:, 1] * U2


class DropBlock2D(nn.Module):
    def __init__(self, drop_prob=0.15, block_size=5):
        super(DropBlock2D, self).__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        if not self.training or self.drop_prob == 0.:
            return x
        gamma = self.drop_prob / \
            (self.block_size ** 2) * \
            (x.shape[-1] ** 2) / ((x.shape[-1] - self.block_size + 1) ** 2)
        mask = (torch.rand(x.shape[0], *x.shape[2:],
                device=x.device) < gamma).float()
        mask = F.max_pool2d(mask, self.block_size, stride=1,
                            padding=self.block_size // 2)
        mask = 1. - mask
        return x * mask.unsqueeze(1) * (mask.numel() / mask.sum())


class ImageClassificationModel(nn.Module):
    def __init__(self, num_classes=100, pretrained=True):
        super(ImageClassificationModel, self).__init__()
        resnet = models.resnet152(
            weights=models.ResNet152_Weights.DEFAULT if pretrained else None)

        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        for param in self.stem.parameters():
            param.requires_grad = False
        for param in self.layer1.parameters():
            param.requires_grad = False

        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.conv1x1_l3 = nn.Sequential(
            nn.Conv2d(1024, 512, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(True))
        self.conv1x1_l4 = nn.Sequential(
            nn.Conv2d(2048, 512, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(True))

        self.sk3 = SKConv(512)
        self.sk4 = SKConv(512)

        self.dropblock = DropBlock2D(
            drop_prob=0.0, block_size=5)  # 預設關閉，由外部排程控制

        # ⭐ 新架構：Transformer + Attention Pooling
        self.transformer = SimpleTransformer(
            dim=512, num_heads=8, feat_size=14)
        self.attn_pool3 = AttentionPooling(in_dim=512)
        self.attn_pool4 = AttentionPooling(in_dim=512)
        self.attn_pool_fused = AttentionPooling(in_dim=512)

        self.emb_l3 = nn.Sequential(nn.BatchNorm1d(
            512), nn.PReLU(), nn.Dropout(p=0.4))
        self.cls_l3 = CosineLinear(512, num_classes)

        self.emb_l4 = nn.Sequential(nn.BatchNorm1d(
            512), nn.PReLU(), nn.Dropout(p=0.4))
        self.cls_l4 = CosineLinear(512, num_classes)

        self.emb_fused = nn.Sequential(
            nn.BatchNorm1d(512), nn.PReLU(), nn.Dropout(p=0.4))
        self.cls_fused = CosineLinear(512, num_classes)

        self._init_weights()

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)

        f3_orig = self.layer3(x)
        f3_c = self.conv1x1_l3(f3_orig)
        f3_sk = self.sk3(f3_c)
        f3_sk = self.dropblock(f3_sk)

        f4_orig = self.layer4(f3_orig)
        f4_c = self.conv1x1_l4(f4_orig)
        f4_sk = self.sk4(f4_c)
        f4_sk = self.dropblock(f4_sk)

        # 全局關係感知
        f_fused = self.transformer(f4_sk)

        # 線性特徵聚合
        p3 = self.attn_pool3(f3_sk)
        p4 = self.attn_pool4(f4_sk)
        p_fused = self.attn_pool_fused(f_fused)

        e3 = self.emb_l3(p3)
        e4 = self.emb_l4(p4)
        e_fused = self.emb_fused(p_fused)

        out3 = self.cls_l3(e3)
        out4 = self.cls_l4(e4)
        out_fused = self.cls_fused(e_fused)

        if self.training:
            return out3, out4, out_fused, e_fused
        else:
            return out_fused

    def get_saliency(self, x):
        # ⭐ 紀錄進入函數前的模型模式 (True = 訓練中, False = 評估中)
        is_training = self.training 
        
        self.eval()
        with torch.no_grad():
            x = self.stem(x)
            x = self.layer1(x)
            x = self.layer2(x)
            f3 = self.layer3(x)
            f4 = self.layer4(f3)
            f4_c = self.conv1x1_l4(f4)
            f4_sk = self.sk4(f4_c)
            # 使用 Transformer 與 Attention Pooling 產生的精準權重圖作為導航
            f_fused = self.transformer(f4_sk)
            attn = self.attn_pool_fused.attn_conv(f_fused)  # [B, 1, H, W]
            saliency = attn.squeeze(1)
            
        # ⭐ 恢復原本的模式，避免破壞推論 (eval) 狀態
        self.train(is_training) 
        return saliency
    
    # 替換掉原本的 freeze_features_for_crt 與 get_classifier_parameters
    def get_finetune_parameters(self):
        """
        回傳兩組參數以供差分學習率 (Differential LR) 使用：
        1. head_params: 分類頭 (CosineLinear) 與 Embedding 層
        2. base_params: 解凍的 Backbone (ResNet layer2~4, SKConv, Transformer)
        """
        head_names = ['emb_l3', 'cls_l3', 'emb_l4', 'cls_l4', 'emb_fused', 'cls_fused']
        head_params = []
        base_params = []
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue  # 跳過固定不動的 stem 與 layer1
                
            if any(x in name for x in head_names):
                head_params.append(param)
            else:
                base_params.append(param)
                
        return head_params, base_params

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def check_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        print(
            f"📊 ResNet152-Transformer-AttnPool (Exp 49) Params: {total/1e6:.2f}M")
        return total < 100_000_000
