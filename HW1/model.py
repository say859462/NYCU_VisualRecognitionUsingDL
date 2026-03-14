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
        """Custom Model with ResNet backbone

        Args:
            num_classes (int, optional): Number of classes for classification. Defaults to 100.
            pretrained (bool, optional): Whether to use pretrained weights. Defaults to True.
        """
        super(ImageClassificationModel, self).__init__()

        # Backbone model : ResNet
        resnet = models.resnet152(
            weights=models.ResNet152_Weights.DEFAULT if pretrained else None,
            replace_stride_with_dilation=[False, False, True])

        self.backbone = nn.Sequential(
            *list(resnet.children())[:-2])
        # in_plane = 2048 for resnet152
        self.cbam = CBAM(in_planes=2048, ratio=16, kernel_size=7)

        # self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.pool = GeM(p=5.0)
        self.flatten = nn.Flatten()

        self.embedding = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(),
            nn.Dropout(p=0.5)
        )

        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.cbam(x)
        x = self.flatten(self.pool(x))
        embeddings = self.embedding(x)
        logits = self.classifier(embeddings)

        return logits

    def check_parameters(self):
        # Check the total number of trainable parameters in the model

        total_params = sum(p.numel()
                           for p in self.parameters())
        print(f"Total trainable parameters: {total_params:,}")

        if total_params < 100_000_000:
            print("Model size is within the 100M limit.")
            return True
        else:
            print("Warning: Model size exceeds the 100M limit!")

        return False
