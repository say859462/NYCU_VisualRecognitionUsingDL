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
    def __init__(self, in_dim, embed_dim, num_classes, num_subcenters=3, dropout=0.2):
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


class GlobalGuidedCrossAttention(nn.Module):
    """Global query retrieves useful detail tokens from multiple granularities."""

    def __init__(self, embed_dim=256, token_dim=256, num_heads=4, token_grid=4, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.token_dim = token_dim
        self.token_grid = token_grid
        self.tokens_per_level = token_grid * token_grid

        self.pool_tokens = nn.AdaptiveAvgPool2d((token_grid, token_grid))

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.kv_proj_l3 = nn.Linear(token_dim, embed_dim)
        self.kv_proj_l4 = nn.Linear(token_dim, embed_dim)

        self.pos_embed_l3 = nn.Parameter(
            torch.zeros(1, self.tokens_per_level, embed_dim))
        self.pos_embed_l4 = nn.Parameter(
            torch.zeros(1, self.tokens_per_level, embed_dim))
        self.granularity_embed = nn.Parameter(torch.zeros(1, 2, embed_dim))

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.fusion_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid(),
        )
        self.out_norm = nn.LayerNorm(embed_dim)
        self.out_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.trunc_normal_(self.pos_embed_l3, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_l4, std=0.02)
        nn.init.trunc_normal_(self.granularity_embed, std=0.02)
        for m in [self.q_proj, self.kv_proj_l3, self.kv_proj_l4]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        for m in self.fusion_gate:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.out_mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def _map_to_tokens(self, feat_map):
        pooled = self.pool_tokens(feat_map)
        return pooled.flatten(2).transpose(1, 2)

    def forward(self, global_embed, feat_l3_proj, feat_l4_proj):
        query = self.q_proj(global_embed).unsqueeze(1)

        tokens_l3 = self.kv_proj_l3(self._map_to_tokens(feat_l3_proj))
        tokens_l4 = self.kv_proj_l4(self._map_to_tokens(feat_l4_proj))

        tokens_l3 = tokens_l3 + self.pos_embed_l3 + \
            self.granularity_embed[:, 0:1, :]
        tokens_l4 = tokens_l4 + self.pos_embed_l4 + \
            self.granularity_embed[:, 1:2, :]
        kv_tokens = torch.cat([tokens_l3, tokens_l4], dim=1)

        attn_output, attn_weights = self.cross_attn(
            query=query,
            key=kv_tokens,
            value=kv_tokens,
            need_weights=True,
            average_attn_weights=False,
        )

        attn_embed = attn_output.squeeze(1)
        gate = self.fusion_gate(torch.cat([global_embed, attn_embed], dim=1))
        fused = global_embed + gate * attn_embed
        fused = self.out_norm(fused)
        fused = fused + self.out_mlp(fused)
        fused = self.out_norm(fused)
        return fused, attn_weights, kv_tokens


class ResidualCorrectionBlock(nn.Module):
    """Use part branches only as correction terms for a semantic backbone."""

    def __init__(self, embed_dim=256, dropout=0.1):
        super().__init__()
        self.part2_delta = nn.Linear(embed_dim, embed_dim)
        self.part4_delta = nn.Linear(embed_dim, embed_dim)

        self.part2_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid(),
        )
        self.part4_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid(),
        )

        self.refine = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )
        self.out_norm = nn.LayerNorm(embed_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        for module in [self.part2_delta, self.part4_delta]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        for gate in [self.part2_gate, self.part4_gate]:
            for m in gate:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)
        for m in self.refine:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, backbone_embed, part2_embed, part4_embed):
        p2_gate = self.part2_gate(
            torch.cat([backbone_embed, part2_embed], dim=1))
        p4_gate = self.part4_gate(
            torch.cat([backbone_embed, part4_embed], dim=1))

        p2_delta = p2_gate * torch.tanh(self.part2_delta(part2_embed))
        p4_delta = p4_gate * torch.tanh(self.part4_delta(part4_embed))

        corrected = backbone_embed + p2_delta + p4_delta
        corrected = corrected + self.refine(corrected)
        corrected = self.out_norm(corrected)
        return corrected, p2_delta, p4_delta, p2_gate, p4_gate


class ImageClassificationModel(nn.Module):
    def __init__(
        self,
        num_classes=100,
        pretrained=True,
        num_subcenters=3,
        embed_dim=256,
    ):
        super().__init__()

        resnet = models.resnet152(
            weights=models.ResNet152_Weights.DEFAULT if pretrained else None
        )

        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
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

        self.gem = GeM(p=3.0, learn_p=True)
        self.pool_2 = nn.AdaptiveAvgPool2d((2, 2))
        self.pool_4_avg = nn.AdaptiveAvgPool2d((4, 4))
        self.pool_4_max = nn.AdaptiveMaxPool2d((4, 4))

        self.global_head = PMGHead(
            512, embed_dim, num_classes, num_subcenters, dropout=0.2
        )
        self.part2_head = PMGHead(
            512 * 4, embed_dim, num_classes, num_subcenters, dropout=0.2
        )
        self.part4_head = PMGHead(
            256 * 16, embed_dim, num_classes, num_subcenters, dropout=0.2
        )

        self.cross_fusion = GlobalGuidedCrossAttention(
            embed_dim=embed_dim,
            token_dim=256,
            num_heads=4,
            token_grid=4,
            dropout=0.1,
        )
        self.correction_fusion = ResidualCorrectionBlock(
            embed_dim=embed_dim,
            dropout=0.1,
        )

        # Final head now treats cross_fused as the semantic backbone.
        self.concat_head = PMGHead(
            embed_dim, embed_dim, num_classes, num_subcenters, dropout=0.2
        )

        self._freeze_shallow_layers()
        self._init_new_layers()

    def _freeze_shallow_layers(self):
        for module in [self.stem, self.layer1]:
            for p in module.parameters():
                p.requires_grad = False

        for module in [
            self.layer2,
            self.layer3,
            self.layer4,
            self.proj_l3,
            self.proj_l4,
            self.fuse,
            self.gem,
            self.global_head,
            self.part2_head,
            self.part4_head,
            self.cross_fusion,
            self.correction_fusion,
            self.concat_head,
        ]:
            for p in module.parameters():
                p.requires_grad = True

    def _init_new_layers(self):
        for module in [self.proj_l3, self.proj_l4, self.fuse]:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    def check_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        print(f"Parameters : {total}")
        return total < 100_000_000

    def get_parameter_groups(self, lr_base):
        head_params = []
        for module in [
            self.proj_l3,
            self.proj_l4,
            self.fuse,
            self.gem,
            self.global_head,
            self.part2_head,
            self.part4_head,
            self.cross_fusion,
            self.correction_fusion,
            self.concat_head,
        ]:
            head_params.extend(
                [p for p in module.parameters() if p.requires_grad])

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
        feat_l2 = self.layer2(x)
        feat_l3 = self.layer3(feat_l2)
        feat_l4 = self.layer4(feat_l3)

        feat_l3_proj = self.proj_l3(feat_l3)
        feat_l4_proj = self.proj_l4(feat_l4)
        feat_l3_proj_down = F.adaptive_avg_pool2d(
            feat_l3_proj, feat_l4_proj.shape[-2:]
        )
        fused_map = self.fuse(
            torch.cat([feat_l3_proj_down, feat_l4_proj], dim=1))

        return feat_l2, feat_l3, feat_l4, feat_l3_proj, feat_l4_proj, fused_map

    def forward_pmg(self, x, return_attn=False):
        _, _, _, feat_l3_proj, feat_l4_proj, fused_map = self.forward_features(
            x)

        global_feat = self.gem(fused_map)
        global_logits, global_embed, global_logits_all = self.global_head(
            global_feat)

        part2_feat = self.pool_2(fused_map).flatten(1)
        part2_logits, part2_embed, part2_logits_all = self.part2_head(
            part2_feat)

        # Fine-detail branch now reads directly from high-resolution L3 features.
        part4_avg = self.pool_4_avg(feat_l3_proj)
        part4_max = self.pool_4_max(feat_l3_proj)
        part4_feat = (part4_avg + part4_max).flatten(1)
        part4_logits, part4_embed, part4_logits_all = self.part4_head(
            part4_feat)

        cross_fused_embed, cross_attn_weights, fine_tokens = self.cross_fusion(
            global_embed, feat_l3_proj, feat_l4_proj
        )
        corrected_embed, part2_delta, part4_delta, part2_gate, part4_gate = self.correction_fusion(
            cross_fused_embed, part2_embed, part4_embed
        )

        concat_logits, concat_embed, concat_logits_all = self.concat_head(
            corrected_embed)

        outputs = {
            "global_logits": global_logits,
            "part2_logits": part2_logits,
            "part4_logits": part4_logits,
            "concat_logits": concat_logits,
            "global_embed": global_embed,
            "part2_embed": part2_embed,
            "part4_embed": part4_embed,
            "cross_fused_embed": cross_fused_embed,
            "corrected_embed": corrected_embed,
            "part2_delta": part2_delta,
            "part4_delta": part4_delta,
            "part2_gate": part2_gate,
            "part4_gate": part4_gate,
            "concat_embed": concat_embed,
            "global_logits_all": global_logits_all,
            "part2_logits_all": part2_logits_all,
            "part4_logits_all": part4_logits_all,
            "concat_logits_all": concat_logits_all,
            "fused_map": fused_map,
            "feat_l3_proj": feat_l3_proj,
            "feat_l4_proj": feat_l4_proj,
            "fine_tokens": fine_tokens,
            "cross_attn_weights": cross_attn_weights,
            "part4_source": feat_l3_proj,
            "part4_avg_map": part4_avg,
            "part4_max_map": part4_max,
        }
        if return_attn:
            return outputs
        return outputs

    def prototype_diversity_loss(self, margin=0.2):
        total_loss = 0.0
        count = 0

        classifiers = [
            self.global_head.classifier,
            self.part2_head.classifier,
            self.part4_head.classifier,
            self.concat_head.classifier,
        ]

        for clf in classifiers:
            if clf.num_subcenters <= 1:
                continue

            w = F.normalize(clf.weight, dim=2)
            _, k, _ = w.shape

            sim = torch.einsum("ckd,cjd->ckj", w, w)
            eye = torch.eye(k, device=sim.device).unsqueeze(0)
            off_diag = sim * (1.0 - eye)

            loss = F.relu(off_diag - margin).mean()
            total_loss = total_loss + loss
            count += 1

        if count == 0:
            return torch.tensor(0.0, device=self.global_head.classifier.weight.device)
        return total_loss / count
