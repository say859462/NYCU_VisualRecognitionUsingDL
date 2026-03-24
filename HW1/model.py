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
    """
    Multi-prototype classifier:
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

        # logits_all: [B, C, K]
        logits_all = torch.einsum("bd,ckd->bck", x, w)
        logits_all = logits_all * self.scale.clamp(min=1.0)

        # class logits: max over sub-centers
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

        # Light bottleneck
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
        self.embedding_fusion = nn.Sequential(
            nn.Linear(512 * 2, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self._freeze_shallow_layers()
        self.set_train_stage(1)
        self._init_new_layers()

    def _freeze_shallow_layers(self):
        for module in [self.stem, self.layer1]:
            for param in module.parameters():
                param.requires_grad = False

    def set_train_stage(self, stage):
        if stage not in (1, 2):
            raise ValueError("stage should be 1 or 2")

        # Stage 1: train layer2~4 + head
        # Stage 2: only train layer4 + head
        for param in self.layer2.parameters():
            param.requires_grad = (stage == 1)
        for param in self.layer3.parameters():
            param.requires_grad = (stage == 1)
        for param in self.layer4.parameters():
            param.requires_grad = True

        for module in [
            self.proj_l3, self.proj_l4, self.fuse,
            self.pool, self.embedding, self.embedding_fusion, self.classifier
        ]:
            for param in module.parameters():
                param.requires_grad = True

    def _head_parameters(self):
        params = []
        for module in [
            self.proj_l3, self.proj_l4, self.fuse,
            self.pool, self.embedding, self.embedding_fusion, self.classifier
        ]:
            params.extend([p for p in module.parameters() if p.requires_grad])
        return params

    def get_parameter_groups(self, lr_base, stage):
        head_params = self._head_parameters()

        if stage == 1:
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

        return [
            {
                "params": [p for p in self.layer4.parameters() if p.requires_grad],
                "lr": lr_base * 0.05,
            },
            {
                "params": head_params,
                "lr": lr_base * 0.5,
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

    def forward_features_with_local(self, x):
        pooled, fused_map = self.forward_features(x)

        # saliency: [B, H, W]
        saliency = fused_map.pow(2).mean(dim=1, keepdim=True)

        # normalize (soft attention)
        saliency = saliency - saliency.amin(dim=(2, 3), keepdim=True)
        saliency = saliency / (saliency.amax(dim=(2, 3), keepdim=True) + 1e-6)

        # soft weighted pooling
        weighted_feat = fused_map * saliency
        local_feat = self.pool(weighted_feat)

        return pooled, local_feat

    def forward_head(self, pooled):
        embed = self.embedding(pooled)
        logits, logits_all = self.classifier(embed)
        return logits, embed, logits_all

    def forward(self, x):
        global_feat, local_feat = self.forward_features_with_local(x)

        fused = torch.cat([global_feat, local_feat], dim=1)
        fused = self.embedding_fusion(fused)

        logits, _ = self.classifier(fused)
        return logits

    def get_saliency(self, x):
        is_training = self.training
        self.eval()
        with torch.no_grad():
            _, fused_map = self.forward_features(x)
            saliency = fused_map.pow(2).mean(dim=1)
        self.train(is_training)
        return saliency

    def prototype_diversity_loss(self, margin=0.2):
        """
        Encourage different sub-centers of the same class
        not to collapse into one prototype.
        """
        w = F.normalize(self.classifier.weight, dim=2)  # [C, K, D]
        c, k, d = w.shape
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
            self.embedding,
            self.embedding_fusion,   # ⭐ 必加
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
                elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        self.classifier.reset_parameters()

    def check_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        print(f"📊 ResNet152-L34Fuse-GeM-SubCenter Params: {total / 1e6:.2f}M")
        return total < 100_000_000
