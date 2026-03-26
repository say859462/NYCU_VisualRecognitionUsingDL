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
    def __init__(self, dim, num_heads=4, mlp_ratio=4.0, dropout=0.2):
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
        if return_attn:
            attn_out, attn_weights = self.attn(
                cls_token,
                tokens,
                tokens,
                need_weights=True,
                average_attn_weights=False
            )
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
        x = F.normalize(x, dim=1)
        w = F.normalize(self.weight, dim=2)

        logits_all = torch.einsum("bd,ckd->bck", x, w)
        logits_all = logits_all * self.scale.clamp(min=1.0)

        class_logits, _ = logits_all.max(dim=2)
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

        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

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

        self.token_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.cross_attn = CrossAttentionBlock(
            dim=512, num_heads=4, mlp_ratio=4.0, dropout=0.1
        )

        self.embedding = nn.Sequential(
            nn.Linear(512, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
        )

        self.classifier = SubCenterClassifier(
            in_features=embed_dim,
            num_classes=num_classes,
            num_subcenters=num_subcenters,
            scale=16.0,
            learn_scale=True
        )

        self._freeze_shallow_layers()
        self._init_new_layers()

    def _freeze_shallow_layers(self):
        for module in [self.stem, self.layer1]:
            for param in module.parameters():
                param.requires_grad = False

        for param in self.layer2.parameters():
            param.requires_grad = True
        for param in self.layer3.parameters():
            param.requires_grad = True
        for param in self.layer4.parameters():
            param.requires_grad = True

        for module in [
            self.proj_l3, self.proj_l4, self.fuse,
            self.pool, self.cross_attn, self.embedding, self.classifier
        ]:
            for param in module.parameters():
                param.requires_grad = True

        self.cls_token.requires_grad = True

    def get_parameter_groups(self, lr_base):
        head_params = []
        for module in [
            self.proj_l3, self.proj_l4, self.fuse,
            self.pool, self.cross_attn, self.embedding, self.classifier
        ]:
            head_params.extend(
                [p for p in module.parameters() if p.requires_grad]
            )

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
        tokens = self.token_pool(fused_map)
        tokens = tokens.flatten(2).transpose(1, 2)
        return tokens

    def cls_cross_attention_from_tokens(self, tokens, return_attn=False):
        batch_size = tokens.size(0)
        cls = self.cls_token.expand(batch_size, -1, -1)

        if return_attn:
            cls, attn_weights = self.cross_attn(
                cls, tokens, return_attn=True
            )
            cls = cls.squeeze(1)
            return cls, attn_weights

        cls = self.cross_attn(cls, tokens, return_attn=False)
        cls = cls.squeeze(1)
        return cls

    def cls_cross_attention_fusion(self, fused_map, return_attn=False):
        tokens = self.build_tokens(fused_map)
        return self.cls_cross_attention_from_tokens(
            tokens, return_attn=return_attn
        )

    def forward_head(self, pooled_512):
        embed = self.embedding(pooled_512)
        logits, logits_all = self.classifier(embed)
        return logits, embed, logits_all

    def forward_view(self, x, return_attn=False):
        _, fused_map = self.forward_features(x)
        tokens = self.build_tokens(fused_map)

        if return_attn:
            cls_feat, attn_weights = self.cls_cross_attention_from_tokens(
                tokens, return_attn=True
            )
        else:
            cls_feat = self.cls_cross_attention_from_tokens(
                tokens, return_attn=False
            )
            attn_weights = None

        logits, embed, logits_all = self.forward_head(cls_feat)
        return {
            "tokens": tokens,
            "cls_feat": cls_feat,
            "logits": logits,
            "embed": embed,
            "logits_all": logits_all,
            "attn_weights": attn_weights
        }

    def forward_with_attention(self, x):
        outputs = self.forward_view(x, return_attn=True)
        return outputs["logits"], outputs["attn_weights"]

    def get_cross_attention_map(self, x):
        is_training = self.training
        self.eval()
        with torch.no_grad():
            _, attn_weights = self.forward_with_attention(x)
            attn_map = attn_weights.mean(dim=1)
            attn_map = attn_map.view(attn_map.size(0), 1, 7, 7)
        self.train(is_training)
        return attn_map

    def forward(self, x):
        outputs = self.forward_view(x, return_attn=False)
        return outputs["logits"]

    def forward_full_local(self, full_x, local1_x, local2_x=None):
        full_outputs = self.forward_view(full_x, return_attn=False)
        local1_outputs = self.forward_view(local1_x, return_attn=False)

        all_tokens = [full_outputs["tokens"], local1_outputs["tokens"]]
        local2_outputs = None
        if local2_x is not None:
            local2_outputs = self.forward_view(local2_x, return_attn=False)
            all_tokens.append(local2_outputs["tokens"])

        fused_tokens = torch.cat(all_tokens, dim=1)
        fused_cls = self.cls_cross_attention_from_tokens(
            fused_tokens, return_attn=False
        )
        fused_logits, fused_embed, fused_logits_all = self.forward_head(
            fused_cls
        )

        return {
            "fused_logits": fused_logits,
            "full_logits": full_outputs["logits"],
            "local1_logits": local1_outputs["logits"],
            "local2_logits": None if local2_outputs is None else local2_outputs["logits"],
            "fused_embed": fused_embed,
            "full_embed": full_outputs["embed"],
            "local1_embed": local1_outputs["embed"],
            "local2_embed": None if local2_outputs is None else local2_outputs["embed"],
            "fused_logits_all": fused_logits_all,
            "full_logits_all": full_outputs["logits_all"],
            "local1_logits_all": local1_outputs["logits_all"],
            "local2_logits_all": None if local2_outputs is None else local2_outputs["logits_all"],
        }

    def get_saliency(self, x):
        is_training = self.training
        self.eval()
        with torch.no_grad():
            _, fused_map = self.forward_features(x)
            saliency = fused_map.pow(2).mean(dim=1)
        self.train(is_training)
        return saliency

    def prototype_diversity_loss(self, margin=0.2):
        w = F.normalize(self.classifier.weight, dim=2)
        _, num_subcenters, _ = w.shape
        if num_subcenters <= 1:
            return torch.tensor(0.0, device=w.device)

        loss = 0.0
        count = 0
        for i in range(num_subcenters):
            for j in range(i + 1, num_subcenters):
                sim = (w[:, i, :] * w[:, j, :]).sum(dim=1)
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
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
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
            f"📊 ResNet152-L34Fuse-GeM-CLS-CrossAttn-SubCenter Params: {total / 1e6:.2f}M"
        )
        return total < 100_000_000