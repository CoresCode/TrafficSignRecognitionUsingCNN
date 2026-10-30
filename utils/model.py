import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    A reusable convolutional block:
    Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU -> MaxPool
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x):
        return self.block(x)


class TrafficSignCNN(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        self.features = nn.Sequential(

            ConvBlock(3, 32),      # 96 -> 48

            ConvBlock(32, 64),     # 48 -> 24

            ConvBlock(64, 128),    # 24 -> 12

            ConvBlock(128, 256)    # 12 -> 6
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(256, 256),

            nn.ReLU(inplace=True),

            nn.Dropout(0.5),

            nn.Linear(256, 128),

            nn.ReLU(inplace=True),

            nn.Dropout(0.3),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):

        x = self.features(x)

        x = self.pool(x)

        x = self.classifier(x)

        return x