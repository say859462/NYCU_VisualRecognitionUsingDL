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


class ImageClassificationModel(nn.Module):
    def __init__(self, num_classes=100, pretrained=True):
        super().__init__()

        resnet = models.resnet152(
            weights=models.ResNet152_Weights.DEFAULT if pretrained else None
        )

        # Pretrained backbone
        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # New layers
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
        self.dropout = nn.Dropout(p=0.2)
        self.classifier = nn.Linear(512, num_classes)

        self._freeze_shallow_layers()
        self.set_train_stage(1)
        self._init_new_layers()  # 只初始化新加的層，不動 pretrained backbone

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

        for module in [self.proj_l3, self.proj_l4, self.fuse, self.pool, self.classifier]:
            for param in module.parameters():
                param.requires_grad = True

    def _head_parameters(self):
        params = []
        for module in [self.proj_l3, self.proj_l4, self.fuse, self.pool, self.classifier]:
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
                "lr": lr_base * 0.1,
            },
            {
                "params": head_params,
                "lr": lr_base,
            },
        ]

    def forward_features(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)

        feat_l3 = self.layer3(x)          # [B, 1024, H, W]
        feat_l4 = self.layer4(feat_l3)    # [B, 2048, H/2, W/2] typically

        feat_l3 = self.proj_l3(feat_l3)   # [B, 256, ...]
        feat_l4 = self.proj_l4(feat_l4)   # [B, 256, ...]

        # Align layer3 to layer4 resolution
        feat_l3 = F.adaptive_avg_pool2d(feat_l3, feat_l4.shape[-2:])

        fused_map = self.fuse(torch.cat([feat_l3, feat_l4], dim=1))  # [B, 512, ...]
        pooled = self.pool(fused_map)
        pooled = self.dropout(pooled)
        return pooled, fused_map

    def forward(self, x):
        pooled, _ = self.forward_features(x)
        return self.classifier(pooled)

    def get_saliency(self, x):
        is_training = self.training
        self.eval()
        with torch.no_grad():
            _, fused_map = self.forward_features(x)
            saliency = fused_map.pow(2).mean(dim=1)
        self.train(is_training)
        return saliency

    def _init_new_layers(self):
        """
        只初始化新加的 head / projection / fusion layers。
        絕對不要重設 pretrained backbone。
        """
        modules_to_init = [
            self.proj_l3,
            self.proj_l4,
            self.fuse,
            self.classifier,
        ]

        for module in modules_to_init:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def check_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        print(f"📊 ResNet152-L34Fuse-GeM Params: {total / 1e6:.2f}M")
        return total < 100_000_000