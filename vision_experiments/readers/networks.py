import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class CropReader(nn.Module):
    def __init__(self, classes, pretrained=True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.features = mobilenet_v3_small(weights=weights).features
        self.readout = nn.Sequential(
            nn.AvgPool2d((3, 2)),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(576 * 3, classes),
        )
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, crop):
        return self.readout(self.features((crop - self.mean) / self.std))


class DistanceReader(nn.Module):
    """Use geometric evidence and low-resolution scene and sign appearance."""

    def __init__(self):
        super().__init__()

        def encoder():
            return nn.Sequential(
                nn.Conv2d(3, 12, 5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(12, 16, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((2, 2)),
                nn.Flatten(),
                nn.Linear(64, 12),
                nn.ReLU(),
            )

        self.scene_encoder = encoder()
        self.sign_encoder = encoder()
        self.geometry = nn.Sequential(
            nn.Linear(13, 48), nn.ReLU(), nn.Linear(48, 32), nn.ReLU()
        )
        self.output = nn.Sequential(
            nn.Linear(56, 36), nn.ReLU(), nn.Dropout(0.1), nn.Linear(36, 3)
        )

    def forward(self, scene, detail, geometry):
        features = torch.cat(
            (
                self.scene_encoder(scene),
                self.sign_encoder(detail),
                self.geometry(geometry),
            ),
            dim=1,
        )
        return self.output(features)
