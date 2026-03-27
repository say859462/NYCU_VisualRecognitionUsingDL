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
    def __init__(self, in_features, num_classes, num_subcenters=3, scale=16.0, learn_scale=True):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.num_subcenters = num_subcenters

        self.weight = nn.Parameter(torch.randn(
            num_classes, num_subcenters, in_features))

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


class LearnableLogitFusion(nn.Module):
    def __init__(self, init_weights=None):
        super().__init__()
        if init_weights is None:
            init_weights = {
                "global": 0.15,
                "part2": 0.25,
                "part4": 0.10,
                "concat": 0.50,
            }

        self.order = ["global", "part2", "part4", "concat"]
        init = torch.tensor([init_weights[k]
                            for k in self.order], dtype=torch.float32)
        init = init / init.sum()
        self.raw_weights = nn.Parameter(torch.log(init.clamp(min=1e-6)))

    def forward(self, logits_dict, enabled_mask=None):
        if enabled_mask is None:
            enabled_mask = {k: True for k in self.order}

        logits_ref = next(iter(logits_dict.values()))
        device = logits_ref.device
        raw = self.raw_weights.to(device)

        mask = torch.tensor(
            [1.0 if enabled_mask.get(k, False) else 0.0 for k in self.order],
            dtype=raw.dtype,
            device=device
        )

        if mask.sum() <= 0:
            raise ValueError("At least one branch must be enabled for fusion.")

        masked_raw = raw.masked_fill(mask == 0, float("-inf"))
        weights = torch.softmax(masked_raw, dim=0)

        fused_logits = 0.0
        for i, k in enumerate(self.order):
            if enabled_mask.get(k, False):
                fused_logits = fused_logits + weights[i] * logits_dict[k]

        return fused_logits, {
            k: float(weights[i].detach().cpu().item())
            for i, k in enumerate(self.order)
        }


class ImageClassificationModel(nn.Module):
    def __init__(
        self,
        num_classes=100,
        pretrained=True,
        num_subcenters=3,
        embed_dim=256,
        fusion_init_weights=None,
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

        self.gem = GeM(p=3.0, learn_p=True)
        self.pool_2 = nn.AdaptiveAvgPool2d((2, 2))
        self.pool_4 = nn.AdaptiveAvgPool2d((4, 4))

        self.global_head = PMGHead(
            512, embed_dim, num_classes, num_subcenters, dropout=0.2)
        self.part2_head = PMGHead(
            512 * 4, embed_dim, num_classes, num_subcenters, dropout=0.2)
        self.part4_head = PMGHead(
            512 * 16, embed_dim, num_classes, num_subcenters, dropout=0.2)
        self.concat_head = PMGHead(
            embed_dim * 3, embed_dim, num_classes, num_subcenters, dropout=0.3)

        self.fusion_head = LearnableLogitFusion(
            init_weights=fusion_init_weights)

        self._freeze_shallow_layers()
        self._init_new_layers()

    def _freeze_shallow_layers(self):
        for module in [self.stem, self.layer1]:
            for p in module.parameters():
                p.requires_grad = False

        for module in [
            self.layer2, self.layer3, self.layer4,
            self.proj_l3, self.proj_l4, self.fuse, self.gem,
            self.global_head, self.part2_head, self.part4_head,
            self.concat_head, self.fusion_head
        ]:
            for p in module.parameters():
                p.requires_grad = True

    def _init_new_layers(self):
        for module in [self.proj_l3, self.proj_l4, self.fuse]:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    def check_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        return total < 100_000_000

    def get_parameter_groups(self, lr_base):
        head_params = []
        for module in [
            self.proj_l3, self.proj_l4, self.fuse, self.gem,
            self.global_head, self.part2_head, self.part4_head,
            self.concat_head, self.fusion_head
        ]:
            head_params.extend(
                [p for p in module.parameters() if p.requires_grad])

        return [
            {"params": [p for p in self.layer2.parameters(
            ) if p.requires_grad], "lr": lr_base * 0.1},
            {"params": [p for p in self.layer3.parameters(
            ) if p.requires_grad], "lr": lr_base * 0.5},
            {"params": [p for p in self.layer4.parameters(
            ) if p.requires_grad], "lr": lr_base * 1.0},
            {"params": head_params, "lr": lr_base * 1.5},
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
        return fused_map

    def get_enabled_branches(self, stage_cfg=None):
        if stage_cfg is None:
            return {"global": True, "part2": True, "part4": True, "concat": True}
        return {
            "global": stage_cfg.get("global_weight", 0.0) > 0,
            "part2": stage_cfg.get("part2_weight", 0.0) > 0,
            "part4": stage_cfg.get("part4_weight", 0.0) > 0,
            "concat": stage_cfg.get("concat_weight", 0.0) > 0,
        }

    def forward_pmg(self, x, stage_cfg=None):
        fused_map = self.forward_features(x)

        global_feat = self.gem(fused_map)
        part2_feat = self.pool_2(fused_map).flatten(1)
        part4_feat = self.pool_4(fused_map).flatten(1)

        global_logits, global_embed, global_logits_all = self.global_head(
            global_feat)
        part2_logits, part2_embed, part2_logits_all = self.part2_head(
            part2_feat)
        part4_logits, part4_embed, part4_logits_all = self.part4_head(
            part4_feat)

        concat_feat = torch.cat(
            [global_embed, part2_embed, part4_embed], dim=1)
        concat_logits, concat_embed, concat_logits_all = self.concat_head(
            concat_feat)

        logits_dict = {
            "global": global_logits,
            "part2": part2_logits,
            "part4": part4_logits,
            "concat": concat_logits,
        }

        enabled_branches = self.get_enabled_branches(stage_cfg)
        fusion_logits, fusion_weights = self.fusion_head(
            logits_dict, enabled_branches)

        return {
            "global_logits": global_logits,
            "part2_logits": part2_logits,
            "part4_logits": part4_logits,
            "concat_logits": concat_logits,
            "fusion_logits": fusion_logits,
            "fusion_weights": fusion_weights,

            "global_embed": global_embed,
            "part2_embed": part2_embed,
            "part4_embed": part4_embed,
            "concat_embed": concat_embed,

            "global_logits_all": global_logits_all,
            "part2_logits_all": part2_logits_all,
            "part4_logits_all": part4_logits_all,
            "concat_logits_all": concat_logits_all,
        }

    def prototype_diversity_loss(self, margin=0.2):
        total_loss = 0.0
        count = 0

        heads = [
            self.global_head.classifier,
            self.part2_head.classifier,
            self.part4_head.classifier,
            self.concat_head.classifier,
        ]

        for clf in heads:
            w = F.normalize(clf.weight, dim=2)
            c, k, d = w.shape
            if k <= 1:
                continue

            sim = torch.einsum("ckd,cjd->ckj", w, w)
            eye = torch.eye(k, device=sim.device).unsqueeze(0)
            off_diag = sim * (1.0 - eye)

            loss = F.relu(off_diag - margin).mean()
            total_loss = total_loss + loss
            count += 1

        if count == 0:
            return torch.tensor(0.0, device=self.global_head.classifier.weight.device)
        return total_loss / count
