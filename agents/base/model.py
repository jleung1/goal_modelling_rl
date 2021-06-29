import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(x)
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return out + x


class Block(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Block, self).__init__()
        self.downsample = nn.Conv2d(input_channels, output_channels, 3, 1, 1)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.res_block1 = ResidualBlock(output_channels)
        self.res_block2 = ResidualBlock(output_channels)

    def forward(self, x):
        x = self.downsample(x)
        x = self.maxpool(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        return x
