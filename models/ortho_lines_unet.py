import torch
import torch.nn as nn
import torch.nn.functional as F


class DeeperDoubleConv(nn.Module):
    """
    A 'triple' or 'quadruple' conv sequence within a single block.
    Example: (Conv->BN->ReLU) x num_convs
    """

    def __init__(self, in_channels, out_channels, num_convs=3):
        super().__init__()
        layers = []
        current_in = in_channels
        for _ in range(num_convs):
            layers.append(nn.Conv2d(current_in, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            current_in = out_channels
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class OrthoLinesUNet(nn.Module):
    """
    ORTHO lines U-Net for finding text lines in images.
    """

    def __init__(
        self, in_channels=3, out_channels=3, base_features=64, num_convs_per_block=3
    ):
        super().__init__()

        # Example: 5 downsampling levels
        features = [
            base_features,
            base_features * 2,
            base_features * 4,
            base_features * 8,
            base_features * 16,
        ]

        self.encoders = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        in_ch = in_channels
        for feat in features:
            self.encoders.append(
                DeeperDoubleConv(in_ch, feat, num_convs=num_convs_per_block)
            )
            in_ch = feat

        # Bottleneck
        self.bottleneck = DeeperDoubleConv(
            features[-1], features[-1] * 2, num_convs=num_convs_per_block
        )

        # Decoder
        self.decoders = nn.ModuleList()
        for feat in reversed(features):
            self.decoders.append(
                nn.ConvTranspose2d(feat * 2, feat, kernel_size=2, stride=2)
            )
            self.decoders.append(
                DeeperDoubleConv(feat * 2, feat, num_convs=num_convs_per_block)
            )

        # Final 1x1 conv
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder
        for encoder in self.encoders:
            x = encoder(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Decoder
        for idx in range(0, len(self.decoders), 2):
            transposed_conv = self.decoders[idx]
            deeper_double_conv = self.decoders[idx + 1]

            x = transposed_conv(x)
            skip_connection = skip_connections[idx // 2]

            # If shape misaligns, pad
            if x.size() != skip_connection.size():
                diffY = skip_connection.size(2) - x.size(2)
                diffX = skip_connection.size(3) - x.size(3)
                x = F.pad(
                    x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
                )

            x = torch.cat((skip_connection, x), dim=1)
            x = deeper_double_conv(x)

        return self.final_conv(x)
