import torch.nn.functional as F
import torch
import torch.nn as nn
from torchvision import models

# CBMA module
# Reference: https://arxiv.org/pdf/1807.06521.pdf


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x

# SEBlock (Squeeze-and-Excitation Block)
# Reference: https://arxiv.org/pdf/1709.01507.pdf
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        # Scale
        return x * y.expand_as(x)

# GeM Pooling Layer(Generalized Mean Pooling)
# Reference: https://arxiv.org/pdf/1711.02512.pdf


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p)


class ImageClassificationModel(nn.Module):
    def __init__(self, num_classes: int = 100, pretrained: bool = True):
        super(ImageClassificationModel, self).__init__()

        # Backbone model : ResNet
        resnet = models.resnet152(
            weights=models.ResNet152_Weights.DEFAULT if pretrained else None
        )

        self.backbone_l1_l3 = nn.Sequential(*list(resnet.children())[:7])
        self.backbone_l4 = nn.Sequential(*list(resnet.children())[7:8])

        # Layer 3 CBAM
        self.se_l3 = SEBlock(in_channels=1024, reduction=16) 

        self.reduce3 = nn.Conv2d(1024, 512, kernel_size=1, bias=False)
        self.reduce4 = nn.Conv2d(2048, 512, kernel_size=1, bias=False)

        self.gem = GeM(p=5)

        self.embedding = nn.Sequential(
            nn.Linear(1024, 768),
            nn.BatchNorm1d(768),
            nn.PReLU(),
            nn.Dropout(p=0.5),

            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(),
            nn.Dropout(p=0.5)
        )
        self.classifier = nn.Linear(512, num_classes)
        self._init_weights()

    def forward(self, x):

        # --- Layer 3 Processing ---
        f3 = self.backbone_l1_l3(x)                        # [B, 1024, 28, 28]
        f3_att = self.se_l3(f3)

        f3_reduced = self.reduce3(f3_att)                  # 1x1 Conv
        p3 = self.gem(f3_reduced).flatten(1)

        # --- Layer 4 Processing ---
        f4 = self.backbone_l4(f3)                          # [B, 2048, 14, 14]
        f4_reduced = self.reduce4(f4)
        p4 = self.gem(f4_reduced).flatten(1)

        # --- Progressive Fusion ---
        fused = torch.cat([p3, p4], dim=1)                 # [B, 1024]
        embeddings = self.embedding(fused)                 # [B, 512]
        return self.classifier(embeddings)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def check_parameters(self):
        # Check the total number of trainable parameters in the model

        total_params = sum(p.numel()
                           for p in self.parameters())
        print(f"Total parameters: {total_params:,}")

        if total_params < 100_000_000:
            print("Model size is within the 100M limit.")
            return True
        else:
            print("Warning: Model size exceeds the 100M limit!")

        return False
