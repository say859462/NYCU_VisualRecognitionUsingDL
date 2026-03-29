import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet152_Weights


class GeM(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6, learn_p: bool = True):
        super().__init__()
        if learn_p:
            self.p = nn.Parameter(torch.ones(1) * p)
        else:
            self.register_buffer("p", torch.tensor([p], dtype=torch.float32))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = torch.clamp(self.p, min=1.0)
        x = x.clamp(min=self.eps).pow(p)
        x = F.adaptive_avg_pool2d(x, 1).pow(1.0 / p)
        return x.flatten(1)


class Res2Adapter(nn.Module):
    def __init__(self, channels: int, scale: int = 4, bottleneck_ratio: int = 4):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.reduce(x)
        splits = torch.split(out, self.split_channels, dim=1)

        outputs = [splits[0]]
        for idx in range(1, self.scale):
            if idx == 1:
                y = self.scale_convs[idx - 1](splits[idx])
            else:
                y = self.scale_convs[idx - 1](splits[idx] + outputs[idx - 1])
            outputs.append(y)

        out = torch.cat(outputs, dim=1)
        out = self.expand(out)
        out = out + identity
        out = self.relu(out)
        return out


class SubCenterClassifier(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        num_subcenters: int = 3,
        scale: float = 16.0,
        learn_scale: bool = True,
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

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, mean=0.0, std=0.01)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.normalize(x, dim=1)
        w = F.normalize(self.weight, dim=2)
        logits_all = torch.einsum("bd,ckd->bck", x, w)
        logits_all = logits_all * self.scale.clamp(min=1.0)
        class_logits, _ = logits_all.max(dim=2)
        return class_logits, logits_all


class PMGHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
        num_classes: int,
        num_subcenters: int = 3,
        dropout: float = 0.2,
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embed = self.proj(x)
        logits, logits_all = self.classifier(embed)
        return logits, embed, logits_all


class PairwiseInteractionFusionHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        num_subcenters: int = 3,
        hidden_ratio: float = 2.0,
        dropout: float = 0.25,
    ):
        super().__init__()
        fusion_dim = embed_dim * 8
        hidden_dim = int(embed_dim * hidden_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
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

    def build_interactions(
        self,
        global_embed: torch.Tensor,
        part2_embed: torch.Tensor,
        part4_embed: torch.Tensor,
    ) -> torch.Tensor:
        features = [
            global_embed,
            part2_embed,
            part4_embed,
            global_embed * part2_embed,
            global_embed * part4_embed,
            part2_embed * part4_embed,
            torch.abs(global_embed - part2_embed),
            torch.abs(global_embed - part4_embed),
        ]
        return torch.cat(features, dim=1)

    def forward(
        self,
        global_embed: torch.Tensor,
        part2_embed: torch.Tensor,
        part4_embed: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fusion_input = self.build_interactions(global_embed, part2_embed, part4_embed)
        fusion_embed = self.mlp(fusion_input)
        fusion_logits, fusion_logits_all = self.classifier(fusion_embed)
        return fusion_logits, fusion_embed, fusion_logits_all, fusion_input


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.10):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = features.device
        features = F.normalize(features, dim=1)
        logits = torch.matmul(features, features.t()) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.t()).float().to(device)
        logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0), device=device)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        positive_count = mask.sum(dim=1)
        valid = positive_count > 0
        if not torch.any(valid):
            return torch.zeros((), device=device, dtype=features.dtype)

        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (positive_count + 1e-8)
        loss = -mean_log_prob_pos[valid].mean()
        return loss


class ImageClassificationModel(nn.Module):
    def __init__(
        self,
        num_classes: int = 100,
        pretrained: bool = True,
        num_subcenters: int = 3,
        embed_dim: int = 256,
        use_logit_router: bool = False,
        router_hidden_dim: int = 256,
        router_dropout: float = 0.1,
        backbone_name: str = "resnet152_partial_res2net",
    ):
        super().__init__()
        del use_logit_router, router_hidden_dim, router_dropout
        if backbone_name != "resnet152_partial_res2net":
            raise ValueError(
                "Only 'resnet152_partial_res2net' is supported in this implementation."
            )
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.embed_dim = embed_dim

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

        self.res2_layer3 = Res2Adapter(1024, scale=4, bottleneck_ratio=4)
        self.res2_layer4 = Res2Adapter(2048, scale=4, bottleneck_ratio=4)

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
        self.concat_head = PairwiseInteractionFusionHead(
            embed_dim=embed_dim,
            num_classes=num_classes,
            num_subcenters=num_subcenters,
            hidden_ratio=2.0,
            dropout=0.25,
        )

        self.supcon_loss = SupervisedContrastiveLoss(temperature=0.10)

        self._freeze_shallow_layers()
        self._init_new_layers()

    def _freeze_shallow_layers(self) -> None:
        for module in [self.stem, self.layer1]:
            for param in module.parameters():
                param.requires_grad = False

    def _init_new_layers(self) -> None:
        for module in [
            self.res2_layer3,
            self.res2_layer4,
            self.global_proj,
            self.part2_proj,
            self.part4_proj,
        ]:
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

    def check_parameters(self) -> bool:
        total = sum(param.numel() for param in self.parameters())
        print(f"Parameters: {total}")
        return total < 100_000_000

    def get_parameter_groups(self, lr_base: float) -> List[Dict[str, object]]:
        head_modules = [
            self.res2_layer3,
            self.res2_layer4,
            self.global_proj,
            self.part2_proj,
            self.part4_proj,
            self.gem,
            self.global_head,
            self.part2_head,
            self.part4_head,
            self.concat_head,
        ]
        head_params = []
        for module in head_modules:
            head_params.extend(
                [param for param in module.parameters() if param.requires_grad]
            )

        return [
            {
                "params": [param for param in self.layer2.parameters() if param.requires_grad],
                "lr": lr_base * 0.1,
            },
            {
                "params": [param for param in self.layer3.parameters() if param.requires_grad],
                "lr": lr_base * 0.5,
            },
            {
                "params": [param for param in self.layer4.parameters() if param.requires_grad],
                "lr": lr_base * 1.0,
            },
            {
                "params": head_params,
                "lr": lr_base * 1.5,
            },
        ]

    def forward_backbone(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        feat_l3 = self.layer3(x)
        feat_l3 = self.res2_layer3(feat_l3)
        feat_l4 = self.layer4(feat_l3)
        feat_l4 = self.res2_layer4(feat_l4)
        return feat_l3, feat_l4

    @staticmethod
    def _normalize_map(x: torch.Tensor) -> torch.Tensor:
        x = x - x.amin(dim=(2, 3), keepdim=True)
        x = x / (x.amax(dim=(2, 3), keepdim=True) + 1e-8)
        return x

    def build_attention_map(
        self,
        global_map: torch.Tensor,
        part2_map: torch.Tensor,
    ) -> torch.Tensor:
        global_sal = self._normalize_map(global_map.mean(dim=1, keepdim=True))
        part2_sal = self._normalize_map(part2_map.mean(dim=1, keepdim=True))
        saliency = 0.5 * global_sal + 0.5 * part2_sal
        return self._normalize_map(saliency)

    def build_background_mask(
        self,
        saliency: torch.Tensor,
        threshold: float = 0.45,
        floor: float = 0.25,
    ) -> torch.Tensor:
        mask = torch.clamp((saliency - threshold) / max(1e-6, 1.0 - threshold), min=0.0, max=1.0)
        return floor + (1.0 - floor) * mask

    def _build_global_feature(self, global_map: torch.Tensor) -> torch.Tensor:
        return self.gem(global_map)

    def _build_part2_feature(self, part2_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        part2_grid = self.pool_2(part2_map)
        return part2_grid.flatten(1), part2_grid

    def _build_part4_feature(self, part4_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        part4_avg = self.pool_4_avg(part4_map)
        part4_max = self.pool_4_max(part4_map)
        part4_grid = 0.5 * (part4_avg + part4_max)
        return part4_grid.flatten(1), part4_grid

    def forward_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat_l3, feat_l4 = self.forward_backbone(x)
        global_map = self.global_proj(feat_l4)
        part2_map = self.part2_proj(feat_l4)
        saliency = self.build_attention_map(global_map, part2_map)

        bg_mask = self.build_background_mask(saliency)
        bg_mask_l3 = F.interpolate(
            bg_mask,
            size=feat_l3.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        part4_source = feat_l3 * bg_mask_l3
        part4_map = self.part4_proj(part4_source)

        return {
            "feat_l3": feat_l3,
            "feat_l4": feat_l4,
            "global_map": global_map,
            "part2_map": part2_map,
            "part4_source": part4_source,
            "part4_map": part4_map,
            "attention_map": saliency,
            "bg_mask": bg_mask,
        }

    def forward_pmg(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = self.forward_features(x)

        global_feat = self._build_global_feature(feats["global_map"])
        part2_feat, part2_grid = self._build_part2_feature(feats["part2_map"])
        part4_feat, part4_grid = self._build_part4_feature(feats["part4_map"])

        global_logits, global_embed, global_logits_all = self.global_head(global_feat)
        part2_logits, part2_embed, part2_logits_all = self.part2_head(part2_feat)
        part4_logits, part4_embed, part4_logits_all = self.part4_head(part4_feat)
        concat_logits, concat_embed, concat_logits_all, fusion_input = self.concat_head(
            global_embed,
            part2_embed,
            part4_embed,
        )

        return {
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
            "part2_grid": part2_grid,
            "part4_grid": part4_grid,
            "fusion_input": fusion_input,
            **feats,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.forward_pmg(x)
        return outputs["concat_logits"]

    def compute_pairwise_ranking_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        margin: float = 0.15,
    ) -> torch.Tensor:
        gt = logits.gather(1, labels.view(-1, 1))
        masked_logits = logits.clone()
        masked_logits.scatter_(1, labels.view(-1, 1), -1e9)
        hardest_neg = masked_logits.max(dim=1, keepdim=True).values
        loss = F.relu(margin - (gt - hardest_neg))
        return loss.mean()

    def compute_stage3_regularization(
        self,
        full_embed: torch.Tensor,
        crop_embed: Optional[torch.Tensor],
        labels: torch.Tensor,
        crop_logits: Optional[torch.Tensor] = None,
        supcon_weight: float = 0.0,
        ranking_weight: float = 0.0,
        ranking_margin: float = 0.15,
    ) -> torch.Tensor:
        device = full_embed.device
        total_loss = torch.zeros((), device=device, dtype=full_embed.dtype)

        if crop_embed is not None and supcon_weight > 0:
            pair_features = torch.cat([full_embed, crop_embed], dim=0)
            pair_labels = torch.cat([labels, labels], dim=0)
            total_loss = total_loss + supcon_weight * self.supcon_loss(pair_features, pair_labels)

        if crop_logits is not None and ranking_weight > 0:
            total_loss = total_loss + ranking_weight * self.compute_pairwise_ranking_loss(
                crop_logits,
                labels,
                margin=ranking_margin,
            )

        return total_loss
