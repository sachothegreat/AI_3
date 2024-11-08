import torch
from torch import nn
import torch.nn.functional as F

# Residual Dense Block for RRDB model
class DenseResidualBlock(nn.Module):
    def __init__(self, filters, res_scale=0.2):
        super(DenseResidualBlock, self).__init__()
        self.res_scale = res_scale

        def block(in_features, non_linearity=True):
            layers = [nn.Conv2d(in_features, filters, kernel_size=3, padding=1)]
            if non_linearity:
                layers += [nn.LeakyReLU(negative_slope=0.2, inplace=True)]
            return nn.Sequential(*layers)

        self.b1 = block(filters)
        self.b2 = block(filters * 2)
        self.b3 = block(filters * 3)
        self.b4 = block(filters * 4)
        self.b5 = block(filters * 5, non_linearity=False)

    def forward(self, x):
        inputs = x
        out1 = self.b1(inputs)
        out2 = self.b2(torch.cat([inputs, out1], 1))
        out3 = self.b3(torch.cat([inputs, out1, out2], 1))
        out4 = self.b4(torch.cat([inputs, out1, out2, out3], 1))
        out5 = self.b5(torch.cat([inputs, out1, out2, out3, out4], 1))
        return out5.mul(self.res_scale) + x

# Residual in Residual Dense Block (RRDB)
class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, filters, res_scale=0.2):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.res_scale = res_scale
        self.dense_blocks = nn.Sequential(
            DenseResidualBlock(filters),
            DenseResidualBlock(filters),
            DenseResidualBlock(filters)
        )

    def forward(self, x):
        return self.dense_blocks(x).mul(self.res_scale) + x

# Generator model using RRDB blocks
class GeneratorRRDB(nn.Module):
    def __init__(self, channels=3, filters=64, num_res_blocks=23, num_upsample=2):
        super(GeneratorRRDB, self).__init__()
        self.conv1 = nn.Conv2d(channels, filters, kernel_size=3, stride=1, padding=1)

        # RRDB Blocks
        self.res_blocks = nn.Sequential(
            *[ResidualInResidualDenseBlock(filters) for _ in range(num_res_blocks)]
        )

        # Second convolutional layer
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)

        # Upsampling layers
        upsample_layers = []
        for _ in range(num_upsample):
            upsample_layers += [
                nn.Conv2d(filters, filters * 4, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.PixelShuffle(upscale_factor=2)
            ]
        self.upsampling = nn.Sequential(*upsample_layers)

        # Final output block
        self.conv3 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(filters, channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)
        return out
