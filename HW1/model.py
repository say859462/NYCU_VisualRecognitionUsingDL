import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


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


class SampleConditionedLogitRouter(nn.Module):
    def __init__(self, num_classes, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.input_dim = num_classes * 4 + 8
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 4),
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    @staticmethod
    def _max_prob_and_gap(logits):
        probs = torch.softmax(logits, dim=1)
        max_prob, _ = probs.max(dim=1, keepdim=True)
        top2 = torch.topk(probs, k=2, dim=1).values
        gap = top2[:, 0:1] - top2[:, 1:2]
        return max_prob, gap

    def build_router_input(
        self,
        global_logits,
        part2_logits,
        part4_logits,
        concat_logits,
    ):
        g_conf, g_gap = self._max_prob_and_gap(global_logits)
        p2_conf, p2_gap = self._max_prob_and_gap(part2_logits)
        p4_conf, p4_gap = self._max_prob_and_gap(part4_logits)
        c_conf, c_gap = self._max_prob_and_gap(concat_logits)

        router_input = torch.cat(
            [
                global_logits.detach(),
                part2_logits.detach(),
                part4_logits.detach(),
                concat_logits.detach(),
                g_conf.detach(),
                p2_conf.detach(),
                p4_conf.detach(),
                c_conf.detach(),
                g_gap.detach(),
                p2_gap.detach(),
                p4_gap.detach(),
                c_gap.detach(),
            ],
            dim=1,
        )

        stats = {
            "global_conf": g_conf.detach(),
            "part2_conf": p2_conf.detach(),
            "part4_conf": p4_conf.detach(),
            "concat_conf": c_conf.detach(),
            "global_gap": g_gap.detach(),
            "part2_gap": p2_gap.detach(),
            "part4_gap": p4_gap.detach(),
            "concat_gap": c_gap.detach(),
        }
        return router_input, stats

    def forward(
        self,
        global_logits,
        part2_logits,
        part4_logits,
        concat_logits,
    ):
        router_input, stats = self.build_router_input(
            global_logits,
            part2_logits,
            part4_logits,
            concat_logits,
        )
        routing_logits = self.mlp(router_input)
        routing_weights = torch.softmax(routing_logits, dim=1)

        wg = routing_weights[:, 0:1]
        wp2 = routing_weights[:, 1:2]
        wp4 = routing_weights[:, 2:3]
        wc = routing_weights[:, 3:4]

        router_logits = (
            wg * global_logits
            + wp2 * part2_logits
            + wp4 * part4_logits
            + wc * concat_logits
        )
        return router_logits, routing_weights, stats


class ImageClassificationModel(nn.Module):
    """
    Res2Net Pure-PMG Realignment v1

    Core idea:
    - global branch: layer4
    - part2 branch: layer4
    - part4 branch: layer3
    - concat head: concat(global_embed, part2_embed, part4_embed)

    This avoids the previous design where all branches came from one fused map.
    """

    def __init__(
        self,
        num_classes=100,
        pretrained=True,
        num_subcenters=3,
        embed_dim=256,
        use_logit_router=False,
        router_hidden_dim=256,
        router_dropout=0.1,
        backbone_name="res2net50_26w_4s",
    ):
        super().__init__()

        self.backbone_name = backbone_name
        self.use_logit_router = use_logit_router

        backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )

        act1 = backbone.act1 if hasattr(backbone, "act1") else backbone.relu
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            act1,
            backbone.maxpool,
        )

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # Res2Net-50/101 bottleneck stages:
        # layer3 -> 1024 channels
        # layer4 -> 2048 channels
        self.global_proj = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.part2_proj = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.part4_proj = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # Kept only for optional compatibility / future visualization hooks.
        # Not used as the main branch generator anymore.
        self.fuse = nn.Identity()

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
            in_dim=512 * 16,
            embed_dim=embed_dim,
            num_classes=num_classes,
            num_subcenters=num_subcenters,
            dropout=0.2,
        )
        self.concat_head = PMGHead(
            in_dim=embed_dim * 3,
            embed_dim=embed_dim,
            num_classes=num_classes,
            num_subcenters=num_subcenters,
            dropout=0.3,
        )

        self.logit_router = None
        if self.use_logit_router:
            self.logit_router = SampleConditionedLogitRouter(
                num_classes=num_classes,
                hidden_dim=router_hidden_dim,
                dropout=router_dropout,
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
            self.global_proj,
            self.part2_proj,
            self.part4_proj,
            self.gem,
            self.global_head,
            self.part2_head,
            self.part4_head,
            self.concat_head,
        ]
        if self.logit_router is not None:
            trainable_modules.append(self.logit_router)

        for module in trainable_modules:
            for param in module.parameters():
                param.requires_grad = True

    def _init_new_layers(self):
        new_modules = [
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
                elif isinstance(submodule, nn.BatchNorm2d):
                    nn.init.ones_(submodule.weight)
                    nn.init.zeros_(submodule.bias)

    def check_parameters(self):
        total = sum(param.numel() for param in self.parameters())
        print(f"Parameters : {total}")
        return total < 100_000_000

    def get_parameter_groups(self, lr_base):
        head_params = []
        head_modules = [
            self.global_proj,
            self.part2_proj,
            self.part4_proj,
            self.gem,
            self.global_head,
            self.part2_head,
            self.part4_head,
            self.concat_head,
        ]
        if self.logit_router is not None:
            head_modules.append(self.logit_router)

        for module in head_modules:
            head_params.extend(
                [param for param in module.parameters() if param.requires_grad]
            )

        return [
            {
                "params": [
                    param for param in self.layer2.parameters()
                    if param.requires_grad
                ],
                "lr": lr_base * 0.1,
            },
            {
                "params": [
                    param for param in self.layer3.parameters()
                    if param.requires_grad
                ],
                "lr": lr_base * 0.5,
            },
            {
                "params": [
                    param for param in self.layer4.parameters()
                    if param.requires_grad
                ],
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
        feat_l4 = self.layer4(feat_l3)
        return feat_l3, feat_l4

    def forward_features(self, x):
        feat_l3, feat_l4 = self.forward_backbone(x)

        global_map = self.global_proj(feat_l4)   # [B, 512, H4, W4]
        part2_map = self.part2_proj(feat_l4)     # [B, 512, H4, W4]
        part4_map = self.part4_proj(feat_l3)     # [B, 512, H3, W3]

        return {
            "feat_l3": feat_l3,
            "feat_l4": feat_l4,
            "global_map": global_map,
            "part2_map": part2_map,
            "part4_map": part4_map,
        }

    def _build_global_feature(self, global_map):
        return self.gem(global_map)

    def _build_part2_feature(self, part2_map):
        part2_grid = self.pool_2(part2_map)  # [B, 512, 2, 2]
        return part2_grid.flatten(1)

    def _build_part4_feature(self, part4_map):
        part4_avg = self.pool_4_avg(part4_map)
        part4_max = self.pool_4_max(part4_map)
        part4_grid = 0.5 * (part4_avg + part4_max)  # part4 AvgMax
        return part4_grid.flatten(1)

    def forward_pmg(self, x):
        feats = self.forward_features(x)

        global_feat = self._build_global_feature(feats["global_map"])
        part2_feat = self._build_part2_feature(feats["part2_map"])
        part4_feat = self._build_part4_feature(feats["part4_map"])

        global_logits, global_embed, global_logits_all = self.global_head(
            global_feat)
        part2_logits, part2_embed, part2_logits_all = self.part2_head(
            part2_feat)
        part4_logits, part4_embed, part4_logits_all = self.part4_head(
            part4_feat)

        concat_input = torch.cat(
            [global_embed, part2_embed, part4_embed],
            dim=1,
        )
        concat_logits, concat_embed, concat_logits_all = self.concat_head(
            concat_input)

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
            "feat_l3": feats["feat_l3"],
            "feat_l4": feats["feat_l4"],
            "global_map": feats["global_map"],
            "part2_map": feats["part2_map"],
            "part4_map": feats["part4_map"],
        }

        if self.logit_router is not None:
            router_logits, router_weights, router_stats = self.logit_router(
                global_logits=global_logits,
                part2_logits=part2_logits,
                part4_logits=part4_logits,
                concat_logits=concat_logits,
            )
            outputs["router_logits"] = router_logits
            outputs["router_weights"] = router_weights
            outputs["router_stats"] = router_stats

        return outputs

    def forward(self, x):
        outputs = self.forward_pmg(x)
        if self.logit_router is not None:
            return outputs["router_logits"]
        return outputs["concat_logits"]

    def prototype_diversity_loss(self):
        # Kept for compatibility with existing train.py
        if self.logit_router is not None:
            device = next(self.parameters()).device
            return torch.zeros(1, device=device, dtype=torch.float32).squeeze(0)

        device = next(self.parameters()).device
        return torch.zeros(1, device=device, dtype=torch.float32).squeeze(0)

    def router_balance_loss(self, router_weights):
        target = torch.full_like(
            router_weights,
            fill_value=1.0 / router_weights.size(1),
        )
        return F.mse_loss(router_weights, target)
