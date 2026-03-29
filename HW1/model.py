import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet152_Weights


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


class SubCenterClassifier(nn.Module):
    def __init__(
        self,
        in_features,
        num_classes,
        num_subcenters=3,
        scale=16.0,
        learn_scale=True,
    ):
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


class PMGHead(nn.Module):
    def __init__(
        self,
        in_dim,
        embed_dim,
        num_classes,
        num_subcenters=3,
        dropout=0.2,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = SubCenterClassifier(
            in_features=embed_dim,
            num_classes=num_classes,
            num_subcenters=num_subcenters,
            scale=16.0,
            learn_scale=True,
        )

    def forward(self, x):
        embed = self.proj(x)
        logits, logits_all = self.classifier(embed)
        return logits, embed, logits_all


class Res2Adapter(nn.Module):
    def __init__(self, channels, scale=4, bottleneck_ratio=4):
        super().__init__()
        assert scale >= 2
        inner_channels = channels // bottleneck_ratio
        assert inner_channels % scale == 0

        self.scale = scale
        self.split_channels = inner_channels // scale

        self.reduce = nn.Sequential(
            nn.Conv2d(channels, inner_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
        )

        self.scale_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    self.split_channels,
                    self.split_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(self.split_channels),
                nn.ReLU(inplace=True),
            )
            for _ in range(scale - 1)
        ])

        self.expand = nn.Sequential(
            nn.Conv2d(inner_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.reduce(x)
        splits = torch.split(out, self.split_channels, dim=1)

        outputs = [splits[0]]
        for i in range(1, self.scale):
            if i == 1:
                y = self.scale_convs[i - 1](splits[i])
            else:
                y = self.scale_convs[i - 1](splits[i] + outputs[i - 1])
            outputs.append(y)

        out = torch.cat(outputs, dim=1)
        out = self.expand(out)
        out = out + identity
        out = self.relu(out)
        return out


class TinyFusionTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads=4, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        attn_input = self.norm1(x)
        attn_out, attn_weights = self.attn(
            attn_input,
            attn_input,
            attn_input,
            need_weights=True,
            average_attn_weights=True,
        )
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attn_weights


class SoftSpatialTokenFusionHead(nn.Module):
    """
    [CLS] + [global] + [part2 4 tokens] + [soft-weighted part4 16 tokens]

    改動重點：
    - 不再 hard top-k selection
    - 改成 soft token weighting
    - 仍保留 branch encoding + 2D positional encoding
    """

    def __init__(
        self,
        embed_dim,
        num_classes,
        num_subcenters=3,
        part2_in_channels=512,
        part4_in_channels=256,
        num_heads=4,
        dropout=0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.global_token_proj = nn.Identity()
        self.part2_token_proj = nn.Linear(part2_in_channels, embed_dim)
        self.part4_token_proj = nn.Linear(part4_in_channels, embed_dim)

        self.part4_token_scorer = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1)
        )

        self.branch_type_embed = nn.Parameter(torch.zeros(1, 4, embed_dim))

        self.cls_pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.global_pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.part2_row_embed = nn.Parameter(torch.zeros(2, embed_dim))
        self.part2_col_embed = nn.Parameter(torch.zeros(2, embed_dim))

        self.part4_row_embed = nn.Parameter(torch.zeros(4, embed_dim))
        self.part4_col_embed = nn.Parameter(torch.zeros(4, embed_dim))

        self.token_dropout = nn.Dropout(dropout)

        self.fusion_block = TinyFusionTransformer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=2.0,
            dropout=dropout,
        )
        self.final_norm = nn.LayerNorm(embed_dim)

        self.classifier = SubCenterClassifier(
            in_features=embed_dim,
            num_classes=num_classes,
            num_subcenters=num_subcenters,
            scale=16.0,
            learn_scale=True,
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.branch_type_embed, std=0.02)

        nn.init.trunc_normal_(self.cls_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.global_pos_embed, std=0.02)

        nn.init.trunc_normal_(self.part2_row_embed, std=0.02)
        nn.init.trunc_normal_(self.part2_col_embed, std=0.02)

        nn.init.trunc_normal_(self.part4_row_embed, std=0.02)
        nn.init.trunc_normal_(self.part4_col_embed, std=0.02)

        nn.init.trunc_normal_(self.part2_token_proj.weight, std=0.02)
        nn.init.zeros_(self.part2_token_proj.bias)

        nn.init.trunc_normal_(self.part4_token_proj.weight, std=0.02)
        nn.init.zeros_(self.part4_token_proj.bias)

        for m in self.part4_token_scorer.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    @staticmethod
    def _build_2d_pos(row_embed, col_embed):
        h, d = row_embed.shape
        w, d2 = col_embed.shape
        assert d == d2
        pos = row_embed[:, None, :] + col_embed[None, :, :]
        return pos.reshape(1, h * w, d)

    def _soft_weight_part4_tokens(self, part4_tokens):
        """
        part4_tokens: [B, 16, D]
        return:
            weighted_tokens: [B, 16, D]
            raw_scores: [B, 16]
            weights: [B, 16] in [0,1]
            preview_topk_idx: [B, 4]
            preview_topk_scores: [B, 4]
        """
        raw_scores = self.part4_token_scorer(part4_tokens).squeeze(-1)   # [B,16]
        weights = torch.sigmoid(raw_scores)                              # [B,16]

        # 不要完全砍掉 token，保留一點 residual，避免 early collapse
        token_scale = 0.20 + 0.80 * weights
        weighted_tokens = part4_tokens * token_scale.unsqueeze(-1)

        preview_topk_scores, preview_topk_idx = torch.topk(
            weights, k=4, dim=1, largest=True, sorted=True
        )
        return weighted_tokens, raw_scores, weights, preview_topk_idx, preview_topk_scores

    def forward(self, global_embed, part2_grid, part4_grid):
        batch_size = global_embed.size(0)

        cls_token = self.cls_token.expand(batch_size, -1, -1)       # [B,1,D]
        global_token = self.global_token_proj(global_embed).unsqueeze(1)

        part2_tokens = part2_grid.flatten(2).transpose(1, 2)        # [B,4,512]
        part2_tokens = self.part2_token_proj(part2_tokens)          # [B,4,D]

        part4_tokens = part4_grid.flatten(2).transpose(1, 2)        # [B,16,256]
        part4_tokens = self.part4_token_proj(part4_tokens)          # [B,16,D]

        cls_type = self.branch_type_embed[:, 0:1, :]
        global_type = self.branch_type_embed[:, 1:2, :]
        part2_type = self.branch_type_embed[:, 2:3, :]
        part4_type = self.branch_type_embed[:, 3:4, :]

        cls_pos = self.cls_pos_embed
        global_pos = self.global_pos_embed
        part2_pos = self._build_2d_pos(self.part2_row_embed, self.part2_col_embed)
        part4_pos = self._build_2d_pos(self.part4_row_embed, self.part4_col_embed)

        cls_token = cls_token + cls_type + cls_pos
        global_token = global_token + global_type + global_pos
        part2_tokens = part2_tokens + part2_type + part2_pos
        part4_tokens = part4_tokens + part4_type + part4_pos

        weighted_part4_tokens, raw_scores, part4_weights, preview_topk_idx, preview_topk_scores = \
            self._soft_weight_part4_tokens(part4_tokens)

        tokens = torch.cat(
            [cls_token, global_token, part2_tokens, weighted_part4_tokens],
            dim=1,
        )  # [B, 1+1+4+16, D]

        tokens = self.token_dropout(tokens)
        fused_tokens, attn_weights = self.fusion_block(tokens)

        cls_embed = self.final_norm(fused_tokens[:, 0, :])
        logits, logits_all = self.classifier(cls_embed)

        aux = {
            "part4_token_scores": raw_scores,
            "part4_token_weights": part4_weights,
            "part4_top4_preview_idx": preview_topk_idx,
            "part4_top4_preview_scores": preview_topk_scores,
        }
        return logits, cls_embed, logits_all, fused_tokens, attn_weights, aux


class ImageClassificationModel(nn.Module):
    def __init__(
        self,
        num_classes=100,
        pretrained=True,
        num_subcenters=3,
        embed_dim=256,
        use_logit_router=False,
        router_hidden_dim=256,
        router_dropout=0.1,
        backbone_name="resnet152_partial_res2net",
    ):
        super().__init__()

        del use_logit_router, router_hidden_dim, router_dropout

        self.backbone_name = backbone_name

        if backbone_name != "resnet152_partial_res2net":
            raise ValueError(
                f"Unsupported backbone_name: {backbone_name}. "
                f"Expected 'resnet152_partial_res2net'."
            )

        weights = ResNet152_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet152(weights=weights)

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.layer3_res2 = nn.Sequential(
            Res2Adapter(1024, scale=4, bottleneck_ratio=4),
            Res2Adapter(1024, scale=4, bottleneck_ratio=4),
        )
        self.layer4_res2 = nn.Sequential(
            Res2Adapter(2048, scale=4, bottleneck_ratio=4),
        )

        self.global_proj = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.part2_proj = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.part4_proj = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.gem = GeM(p=3.0, learn_p=True)
        self.pool_2 = nn.AdaptiveAvgPool2d((2, 2))
        self.pool_4_avg = nn.AdaptiveAvgPool2d((4, 4))
        self.pool_4_max = nn.AdaptiveMaxPool2d((4, 4))

        self.global_head = PMGHead(
            in_dim=512,
            embed_dim=embed_dim,
            num_classes=num_classes,
            num_subcenters=num_subcenters,
            dropout=0.2,
        )
        self.part2_head = PMGHead(
            in_dim=512 * 4,
            embed_dim=embed_dim,
            num_classes=num_classes,
            num_subcenters=num_subcenters,
            dropout=0.2,
        )
        self.part4_head = PMGHead(
            in_dim=256 * 16,
            embed_dim=embed_dim,
            num_classes=num_classes,
            num_subcenters=num_subcenters,
            dropout=0.2,
        )

        self.concat_head = SoftSpatialTokenFusionHead(
            embed_dim=embed_dim,
            num_classes=num_classes,
            num_subcenters=num_subcenters,
            part2_in_channels=512,
            part4_in_channels=256,
            num_heads=4,
            dropout=0.1,
        )

        self._freeze_shallow_layers()
        self._init_new_layers()

    def _freeze_shallow_layers(self):
        for module in [self.stem, self.layer1]:
            for param in module.parameters():
                param.requires_grad = False

        trainable_modules = [
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer3_res2,
            self.layer4_res2,
            self.global_proj,
            self.part2_proj,
            self.part4_proj,
            self.gem,
            self.global_head,
            self.part2_head,
            self.part4_head,
            self.concat_head,
        ]

        for module in trainable_modules:
            for param in module.parameters():
                param.requires_grad = True

    def _init_new_layers(self):
        new_modules = [
            self.layer3_res2,
            self.layer4_res2,
            self.global_proj,
            self.part2_proj,
            self.part4_proj,
        ]
        for module in new_modules:
            for submodule in module.modules():
                if isinstance(submodule, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        submodule.weight,
                        mode="fan_out",
                        nonlinearity="relu",
                    )
                elif isinstance(submodule, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    nn.init.ones_(submodule.weight)
                    nn.init.zeros_(submodule.bias)

    def check_parameters(self):
        total = sum(param.numel() for param in self.parameters())
        print(f"Parameters : {total}")
        return total < 100_000_000

    def get_parameter_groups(self, lr_base):
        head_params = []
        head_modules = [
            self.layer3_res2,
            self.layer4_res2,
            self.global_proj,
            self.part2_proj,
            self.part4_proj,
            self.gem,
            self.global_head,
            self.part2_head,
            self.part4_head,
            self.concat_head,
        ]
        for module in head_modules:
            head_params.extend(
                [param for param in module.parameters() if param.requires_grad]
            )

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

    def forward_backbone(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        feat_l2 = self.layer2(x)
        feat_l3 = self.layer3(feat_l2)
        feat_l3 = self.layer3_res2(feat_l3)
        feat_l4 = self.layer4(feat_l3)
        feat_l4 = self.layer4_res2(feat_l4)
        return feat_l2, feat_l3, feat_l4

    def forward_features(self, x):
        feat_l2, feat_l3, feat_l4 = self.forward_backbone(x)

        global_map = self.global_proj(feat_l4)
        part2_map = self.part2_proj(feat_l3)
        part4_map = self.part4_proj(feat_l2)

        return {
            "feat_l2": feat_l2,
            "feat_l3": feat_l3,
            "feat_l4": feat_l4,
            "global_map": global_map,
            "part2_map": part2_map,
            "part4_map": part4_map,
        }

    def _build_global_feature(self, global_map):
        return self.gem(global_map)

    def _build_part2_grid(self, part2_map):
        return self.pool_2(part2_map)

    def _build_part4_grid(self, part4_map):
        part4_avg = self.pool_4_avg(part4_map)
        part4_max = self.pool_4_max(part4_map)
        return 0.5 * (part4_avg + part4_max)

    def forward_pmg(self, x):
        feats = self.forward_features(x)

        global_feat = self._build_global_feature(feats["global_map"])
        part2_grid = self._build_part2_grid(feats["part2_map"])
        part4_grid = self._build_part4_grid(feats["part4_map"])

        part2_feat = part2_grid.flatten(1)
        part4_feat = part4_grid.flatten(1)

        global_logits, global_embed, global_logits_all = self.global_head(global_feat)
        part2_logits, part2_embed, part2_logits_all = self.part2_head(part2_feat)
        part4_logits, part4_embed, part4_logits_all = self.part4_head(part4_feat)

        (
            concat_logits,
            concat_embed,
            concat_logits_all,
            fusion_tokens,
            fusion_attn,
            fusion_aux,
        ) = self.concat_head(global_embed, part2_grid, part4_grid)

        outputs = {
            "global_logits": global_logits,
            "part2_logits": part2_logits,
            "part4_logits": part4_logits,
            "concat_logits": concat_logits,

            "global_embed": global_embed,
            "part2_embed": part2_embed,
            "part4_embed": part4_embed,
            "concat_embed": concat_embed,

            "global_logits_all": global_logits_all,
            "part2_logits_all": part2_logits_all,
            "part4_logits_all": part4_logits_all,
            "concat_logits_all": concat_logits_all,

            "feat_l2": feats["feat_l2"],
            "feat_l3": feats["feat_l3"],
            "feat_l4": feats["feat_l4"],
            "global_map": feats["global_map"],
            "part2_map": feats["part2_map"],
            "part4_map": feats["part4_map"],

            "part2_grid": part2_grid,
            "part4_grid": part4_grid,

            "fusion_tokens": fusion_tokens,
            "fusion_attn": fusion_attn,

            "part4_token_scores": fusion_aux["part4_token_scores"],
            "part4_token_weights": fusion_aux["part4_token_weights"],
            "part4_top4_preview_idx": fusion_aux["part4_top4_preview_idx"],
            "part4_top4_preview_scores": fusion_aux["part4_top4_preview_scores"],
        }
        return outputs

    def forward(self, x):
        outputs = self.forward_pmg(x)
        return outputs["concat_logits"]

    def prototype_diversity_loss(self):
        device = next(self.parameters()).device
        return torch.zeros(1, device=device, dtype=torch.float32).squeeze(0)