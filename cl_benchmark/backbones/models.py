# cl_benchmark/backbones/models.py
import torch
import torch.nn as nn
from torchvision import models
from collections import OrderedDict


# =========================================================
# BaseModel (provides expand_output_layer, classifier logic)
# =========================================================
class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Identity()
        self.fc_input_features = 0
        self.classifier = None
        self.num_classes = 0

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)

        # Lazy-initialize classifier on first forward
        if self.classifier is None and self.fc_input_features == 0:
            self.fc_input_features = x.size(1)
            self.classifier = nn.Linear(
                self.fc_input_features, max(1, self.num_classes)
            ).to(x.device)

        if self.classifier is None:
            raise RuntimeError(
                "Classifier not initialized; call expand_output_layer() first."
            )

        return self.classifier(x)

    def expand_output_layer(self, new_classes):
        """Add new output nodes for incremental tasks."""
        if self.classifier is None or self.fc_input_features == 0:
            self.num_classes += new_classes
            return

        old = self.num_classes
        self.num_classes += new_classes

        old_w = self.classifier.weight.data.clone()
        old_b = self.classifier.bias.data.clone()

        new_fc = nn.Linear(self.fc_input_features, self.num_classes).to(old_w.device)

        new_fc.weight.data[:old] = old_w
        new_fc.bias.data[:old] = old_b

        self.classifier = new_fc


# =========================================================
# SimpleCNN
# =========================================================
class SimpleCNN(BaseModel):
    def __init__(self, input_channels=1):
        super().__init__()
        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(input_channels, 32, 3, padding=1)),
                    ("relu1", nn.ReLU()),
                    ("pool1", nn.MaxPool2d(2)),
                    ("conv2", nn.Conv2d(32, 64, 3, padding=1)),
                    ("relu2", nn.ReLU()),
                    ("pool2", nn.MaxPool2d(2)),
                    ("gap", nn.AdaptiveAvgPool2d((1, 1))),
                ]
            )
        )


# =========================================================
# ResNet18 (no pretrained weights)
# =========================================================
class ResNet18(BaseModel):
    def __init__(self, input_channels=3):
        super().__init__()
        net = models.resnet18(weights=None)

        if input_channels == 1:
            net.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
            net.maxpool = nn.Identity()

        self.features = nn.Sequential(*list(net.children())[:-1])
        self.fc_input_features = net.fc.in_features


# =========================================================
# ResNet18_pretrained (ImageNet)
# =========================================================
class ResNet18_pretrained(BaseModel):
    def __init__(self, input_channels=3):
        super().__init__()

        # Load pretrained
        try:
            weights = models.ResNet18_Weights.DEFAULT
            net = models.resnet18(weights=weights)
        except Exception:
            net = models.resnet18(pretrained=True)

        # Fix first conv if grayscale
        if input_channels == 1:
            net.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        self.features = nn.Sequential(*list(net.children())[:-1])

        self.fc_input_features = net.fc.in_features
        self.num_classes = 0
        self.classifier = None

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)

        # lazy init if needed
        if self.classifier is None:
            self.classifier = nn.Linear(
                self.fc_input_features, max(1, self.num_classes)
            ).to(x.device)

        return self.classifier(x)


# =========================================================
# ViTExtractor (patch-level features)
# =========================================================
class ViTExtractor(nn.Module):
    def __init__(self, pretrained=True, image_size=224, patch_size=16, device="cpu"):
        super().__init__()

        # load ViT from torchvision
        try:
            if pretrained:
                weights = models.ViT_B_16_Weights.DEFAULT
                vit = models.vit_b_16(weights=weights)
            else:
                vit = models.vit_b_16(weights=None)
        except Exception:
            vit = models.vit_b_16(pretrained=pretrained)

        # Extract components
        self.patch_embed = (
            vit._modules["conv_proj"] if "conv_proj" in vit._modules else vit.conv_proj
        )
        self.encoder = vit.encoder

        self.pre_logits = vit._modules.get("pre_logits", nn.Identity())
        self.hidden_dim = (
            vit.hidden_dim if hasattr(vit, "hidden_dim") else vit.heads.embed_dim
        )

        self.num_patches = (image_size // patch_size) ** 2

        self.to(device)

    def forward(self, x):
        # conv patch projection
        x = self.patch_embed(x)
        B, C, H, W = x.shape

        # flatten patches
        x = x.flatten(2).transpose(1, 2)  # (B, N_patches, hidden_dim)

        # create dummy CLS token
        cls_token = torch.zeros(B, 1, x.size(2), device=x.device)
        x = torch.cat([cls_token, x], dim=1)

        # encoder forward
        out = self.encoder(x)

        # drop CLS token
        patches = out[:, 1:, :].contiguous()
        return patches
