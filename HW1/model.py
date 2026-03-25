import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6, learn_p=True):
        super().__init__()
        if learn_p:
            self.p = nn.Parameter(torch.ones(1) * p)
        else:
            self.register_buffer("p", torch.tensor([p], dtype=torch.float32))
        self.eps = eps

    def forward(self, x):
        p = torch.clamp(self.p, min=1.0)
        x = x.clamp(min=self.eps).pow(p)
        x = F.adaptive_avg_pool2d(x, 1).pow(1.0 / p)
        return x.flatten(1)


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)

        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, cls_token, tokens, return_attn=False):
        # cls_token: [B, 1, C]
        # tokens:    [B, N, C]
        if return_attn:
            attn_out, attn_weights = self.attn(
                cls_token,
                tokens,
                tokens,
                need_weights=True,
                average_attn_weights=False
            )
            # attn_weights: [B, num_heads, 1, N]
        else:
            attn_out, _ = self.attn(
                cls_token,
                tokens,
                tokens,
                need_weights=False
            )
            attn_weights = None

        cls_token = self.norm1(cls_token + attn_out)

        mlp_out = self.mlp(cls_token)
        cls_token = self.norm2(cls_token + mlp_out)

        if return_attn:
            return cls_token, attn_weights
        return cls_token


class SubCenterClassifier(nn.Module):
    """
    每個 class 有 K 個 sub-centers，logit 取 max over sub-centers
    """

    def __init__(self, in_features, num_classes, num_subcenters=3, scale=16.0, learn_scale=True):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.num_subcenters = num_subcenters

        self.weight = nn.Parameter(
            torch.randn(num_classes, num_subcenters, in_features)
        )

        if learn_scale:
            self.scale = nn.Parameter(torch.tensor(float(scale)))
        else:
            self.register_buffer("scale", torch.tensor(float(scale)))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, mean=0.0, std=0.01)

    def forward(self, x):
        # x: [B, D]
        x = F.normalize(x, dim=1)
        w = F.normalize(self.weight, dim=2)  # [C, K, D]

        logits_all = torch.einsum("bd,ckd->bck", x, w)  # [B, C, K]
        logits_all = logits_all * self.scale.clamp(min=1.0)

        class_logits, _ = logits_all.max(dim=2)  # [B, C]
        return class_logits, logits_all


class ImageClassificationModel(nn.Module):
    def __init__(
        self,
        num_classes=100,
        pretrained=True,
        num_subcenters=3,
        embed_dim=256
    ):
        super().__init__()

        resnet = models.resnet152(
            weights=models.ResNet152_Weights.DEFAULT if pretrained else None
        )

        # Backbone
        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # L3/L4 fusion
        self.proj_l3 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.proj_l4 = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.pool = GeM(p=3.0, learn_p=True)

        # CLS token + Cross-Attention fusion
        self.token_pool = nn.AdaptiveAvgPool2d((4, 4))  # 16 tokens
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.cross_attn = CrossAttentionBlock(
            dim=512, num_heads=4, mlp_ratio=4.0, dropout=0.1)

        # Embedding head
        self.embedding = nn.Sequential(
            nn.Linear(512, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
        )

        # Multi-prototype classifier
        self.classifier = SubCenterClassifier(
            in_features=embed_dim,
            num_classes=num_classes,
            num_subcenters=num_subcenters,
            scale=16.0,
            learn_scale=True
        )

        self._freeze_shallow_layers()
        self.set_train_stage(1)
        self._init_new_layers()

    # =========================
    # training stage control
    # =========================
    def _freeze_shallow_layers(self):
        for module in [self.stem, self.layer1]:
            for param in module.parameters():
                param.requires_grad = False

    def set_train_stage(self, stage):
        if stage not in (1, 2):
            raise ValueError("stage should be 1 or 2")

        if stage == 1:
            # backbone
            for param in self.layer2.parameters():
                param.requires_grad = True
            for param in self.layer3.parameters():
                param.requires_grad = True
            for param in self.layer4.parameters():
                param.requires_grad = True

            # fusion / attention / head
            for module in [
                self.proj_l3, self.proj_l4, self.fuse,
                self.pool, self.cross_attn, self.embedding, self.classifier
            ]:
                for param in module.parameters():
                    param.requires_grad = True

            self.cls_token.requires_grad = True

        else:  # stage == 2
            # 完全凍結 backbone 與 fusion / attention
            for param in self.layer2.parameters():
                param.requires_grad = False
            for param in self.layer3.parameters():
                param.requires_grad = False
            for param in self.layer4.parameters():
                param.requires_grad = False

            for module in [self.proj_l3, self.proj_l4, self.fuse, self.pool, self.cross_attn]:
                for param in module.parameters():
                    param.requires_grad = False

            self.cls_token.requires_grad = False

            # 只保留 classifier head
            for param in self.embedding.parameters():
                param.requires_grad = True
            for param in self.classifier.parameters():
                param.requires_grad = True

    def _head_parameters(self):
        params = []

        for module in [self.embedding, self.classifier]:
            params.extend([p for p in module.parameters() if p.requires_grad])

        return params

    def get_parameter_groups(self, lr_base, stage):
        if stage == 1:
            head_params = []
            for module in [
                self.proj_l3, self.proj_l4, self.fuse,
                self.pool, self.cross_attn, self.embedding, self.classifier
            ]:
                head_params.extend([p for p in module.parameters() if p.requires_grad])

            if self.cls_token.requires_grad:
                head_params.append(self.cls_token)

            return [
                {
                    "params": [p for p in self.layer2.parameters() if p.requires_grad],
                    "lr": lr_base * 0.1,
                },
                {
                    "params": [p for p in self.layer3.parameters() if p.requires_grad],
                    "lr": lr_base * 0.5,
                },
                {
                    "params": [p for p in self.layer4.parameters() if p.requires_grad],
                    "lr": lr_base * 1.0,
                },
                {
                    "params": head_params,
                    "lr": lr_base * 1.5,
                },
            ]

        # stage 2: head-only
        head_params = self._head_parameters()
        return [
            {
                "params": head_params,
                "lr": lr_base,
            }
        ]

    # =========================
    # feature extraction
    # =========================
    def forward_features(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)

        feat_l3 = self.layer3(x)
        feat_l4 = self.layer4(feat_l3)

        feat_l3 = self.proj_l3(feat_l3)
        feat_l4 = self.proj_l4(feat_l4)
        feat_l3 = F.adaptive_avg_pool2d(feat_l3, feat_l4.shape[-2:])

        fused_map = self.fuse(torch.cat([feat_l3, feat_l4], dim=1))
        pooled = self.pool(fused_map)
        return pooled, fused_map

    def build_tokens(self, fused_map):
        # fused_map: [B, 512, H, W]
        tokens = self.token_pool(fused_map)          # [B, 512, 4, 4]
        tokens = tokens.flatten(2).transpose(1, 2)   # [B, 16, 512]
        return tokens

    def forward_with_attention(self, x):
        _, fused_map = self.forward_features(x)
        cls_feat, attn_weights = self.cls_cross_attention_fusion(
            fused_map, return_attn=True
        )
        logits, _, _ = self.forward_head(cls_feat)
        return logits, attn_weights

    def get_cross_attention_map(self, x):
        """
        回傳 CLS token 對 local tokens 的 attention map
        output: [B, 1, 4, 4]
        """
        self.eval()
        with torch.no_grad():
            _, attn_weights = self.forward_with_attention(x)
            # attn_weights: [B, num_heads, 1, 16]
            attn_map = attn_weights.mean(dim=1)       # [B, 1, 16]
            attn_map = attn_map.view(attn_map.size(0), 1, 4, 4)
        return attn_map

    def cls_cross_attention_fusion(self, fused_map, return_attn=False):
        tokens = self.build_tokens(fused_map)          # [B, 16, 512]
        B = tokens.size(0)
        cls = self.cls_token.expand(B, -1, -1)        # [B, 1, 512]

        if return_attn:
            cls, attn_weights = self.cross_attn(cls, tokens, return_attn=True)
            cls = cls.squeeze(1)                      # [B, 512]
            return cls, attn_weights
        else:
            cls = self.cross_attn(cls, tokens, return_attn=False)
            cls = cls.squeeze(1)                      # [B, 512]
            return cls

    def forward_head(self, pooled_512):
        embed = self.embedding(pooled_512)
        logits, logits_all = self.classifier(embed)
        return logits, embed, logits_all

    # =========================
    # main forward
    # =========================
    def forward(self, x):
        _, fused_map = self.forward_features(x)
        cls_feat = self.cls_cross_attention_fusion(fused_map)
        logits, _, _ = self.forward_head(cls_feat)
        return logits

    # =========================
    # utilities for train / vis
    # =========================
    def get_saliency(self, x):
        is_training = self.training
        self.eval()
        with torch.no_grad():
            _, fused_map = self.forward_features(x)
            saliency = fused_map.pow(2).mean(dim=1)
        self.train(is_training)
        return saliency

    def prototype_diversity_loss(self, margin=0.2):
        w = F.normalize(self.classifier.weight, dim=2)  # [C, K, D]
        _, k, _ = w.shape
        if k <= 1:
            return torch.tensor(0.0, device=w.device)

        loss = 0.0
        count = 0
        for i in range(k):
            for j in range(i + 1, k):
                sim = (w[:, i, :] * w[:, j, :]).sum(dim=1)  # [C]
                loss = loss + F.relu(sim - margin).mean()
                count += 1

        return loss / max(count, 1)

    def _init_new_layers(self):
        modules_to_init = [
            self.proj_l3,
            self.proj_l4,
            self.fuse,
            self.cross_attn,
            self.embedding,
        ]

        for module in modules_to_init:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
                    if hasattr(m, "weight") and m.weight is not None:
                        nn.init.constant_(m.weight, 1)
                    if hasattr(m, "bias") and m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        self.classifier.reset_parameters()

    def check_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        print(
            f"📊 ResNet152-L34Fuse-GeM-CLS-CrossAttn-SubCenter Params: {total / 1e6:.2f}M")
        return total < 100_000_000
