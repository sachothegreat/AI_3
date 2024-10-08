import torch
from torch import nn
import torch.nn.functional as F

# Define the Residual Block for Generator
class ResidualBlock(nn.Module):
    def __init__(self, filters=64):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(filters)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.3)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(filters)

    def forward(self, x):
        residual = x
        x = self.leaky_relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        return x + residual

# Define the Generator (ESRGAN-like)
class Generator(nn.Module):
    def __init__(self, num_residual_blocks=16):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4)
        self.bn1 = nn.BatchNorm2d(64)
        self.leaky_relu = nn.LeakyReLU(0.2)

        # Residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_residual_blocks)]
        )

        # Upsampling layers
        self.upsample1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upsample2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # Final output layer
        self.conv2 = nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        x = self.leaky_relu(self.bn1(self.conv1(x)))
        x = self.residual_blocks(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.leaky_relu(self.upsample1(x))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.sigmoid(self.conv2(x))
        return x
