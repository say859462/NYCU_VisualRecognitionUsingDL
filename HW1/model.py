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
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    @staticmethod
    def _max_prob_and_gap(logits):
        probs = torch.softmax(logits, dim=1)
        max_prob, _ = probs.max(dim=1, keepdim=True)
        top2 = torch.topk(probs, k=2, dim=1).values
        gap = top2[:, 0:1] - top2[:, 1:2]
        return max_prob, gap

    def build_router_input(self, global_logits, part2_logits, part4_logits, concat_logits):
        g_conf, g_gap = self._max_prob_and_gap(global_logits)
        p2_conf, p2_gap = self._max_prob_and_gap(part2_logits)
        p4_conf, p4_gap = self._max_prob_and_gap(part4_logits)
        c_conf, c_gap = self._max_prob_and_gap(concat_logits)
        router_input = torch.cat([
            global_logits.detach(),
            part2_logits.detach(),
            part4_logits.detach(),
            concat_logits.detach(),
            g_conf.detach(), p2_conf.detach(), p4_conf.detach(), c_conf.detach(),
            g_gap.detach(), p2_gap.detach(), p4_gap.detach(), c_gap.detach(),
        ], dim=1)
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

    def forward(self, global_logits, part2_logits, part4_logits, concat_logits):
        router_input, stats = self.build_router_input(
            global_logits, part2_logits, part4_logits, concat_logits
        )
        routing_logits = self.mlp(router_input)
        routing_weights = torch.softmax(routing_logits, dim=1)

        wg = routing_weights[:, 0:1]
        wp2 = routing_weights[:, 1:2]
        wp4 = routing_weights[:, 2:3]
        wc = routing_weights[:, 3:4]

        router_logits = (
            wg * global_logits +
            wp2 * part2_logits +
            wp4 * part4_logits +
            wc * concat_logits
        )
        return router_logits, routing_weights, stats


class ImageClassificationModel(nn.Module):
    def __init__(
        self,
        num_classes=100,
        pretrained=True,
        num_subcenters=3,
        embed_dim=256,
        use_logit_router=True,
        router_hidden_dim=256,
        router_dropout=0.1,
    ):
        super().__init__()

        resnet = models.resnet152(
            weights=models.ResNet152_Weights.DEFAULT if pretrained else None
        )

        self.use_logit_router = use_logit_router

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
            512, embed_dim, num_classes, num_subcenters, dropout=0.2)
        self.part2_head = PMGHead(
            512 * 4, embed_dim, num_classes, num_subcenters, dropout=0.2)
        self.part4_head = PMGHead(
            512 * 16, embed_dim, num_classes, num_subcenters, dropout=0.2)
        self.concat_head = PMGHead(
            embed_dim * 3, embed_dim, num_classes, num_subcenters, dropout=0.3)

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
            for p in module.parameters():
                p.requires_grad = False

        trainable_modules = [
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
            self.concat_head,
        ]
        if self.logit_router is not None:
            trainable_modules.append(self.logit_router)

        for module in trainable_modules:
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
        print(f"Parameters : {total}")
        return total < 100_000_000

    def get_parameter_groups(self, lr_base):
        head_params = []
        modules = [
            self.proj_l3,
            self.proj_l4,
            self.fuse,
            self.gem,
            self.global_head,
            self.part2_head,
            self.part4_head,
            self.concat_head,
        ]
        if self.logit_router is not None:
            modules.append(self.logit_router)

        for module in modules:
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
        feat_l2 = self.layer2(x)
        feat_l3 = self.layer3(feat_l2)
        feat_l4 = self.layer4(feat_l3)

        feat_l3_proj = self.proj_l3(feat_l3)
        feat_l4_proj = self.proj_l4(feat_l4)
        feat_l3_proj_down = F.adaptive_avg_pool2d(
            feat_l3_proj, feat_l4_proj.shape[-2:])
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

        part4_avg = self.pool_4_avg(fused_map)
        part4_max = self.pool_4_max(fused_map)
        part4_feat = (part4_avg + part4_max).flatten(1)
        part4_logits, part4_embed, part4_logits_all = self.part4_head(
            part4_feat)

        concat_feat = torch.cat(
            [global_embed, part2_embed, part4_embed], dim=1)
        concat_logits, concat_embed, concat_logits_all = self.concat_head(
            concat_feat)

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
            "fused_map": fused_map,
            "feat_l3_proj": feat_l3_proj,
            "feat_l4_proj": feat_l4_proj,
            "part4_source": fused_map,
            "part4_avg_map": part4_avg,
            "part4_max_map": part4_max,
        }

        if self.logit_router is not None:
            router_logits, router_weights, router_stats = self.logit_router(
                global_logits, part2_logits, part4_logits, concat_logits
            )
            outputs["router_logits"] = router_logits
            outputs["router_weights"] = router_weights
            outputs.update({f"router_{k}": v for k, v in router_stats.items()})

        return outputs

    def router_balance_loss(self, router_weights, target_prior=None):
        if target_prior is None:
            target_prior = torch.full(
                (router_weights.size(1),),
                1.0 / router_weights.size(1),
                device=router_weights.device,
            )
        mean_w = router_weights.mean(dim=0)
        return F.kl_div(mean_w.clamp_min(1e-8).log(), target_prior, reduction="batchmean")

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
