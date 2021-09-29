import numpy as np

import torch
import torch.nn as nn


class XuNet(nn.Module):
    def __init__(self):
        super(XuNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8,
                               kernel_size=5, padding=2, bias=None)
        self.norm1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16,
                               kernel_size=5, padding=2, bias=None)
        self.norm2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32,
                               kernel_size=1, bias=None)
        self.norm3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=1, bias=None)
        self.norm4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=1,  bias=None)
        self.norm5 = nn.BatchNorm2d(128)
        self.pool = nn.AvgPool2d(kernel_size=5, stride=2, padding=2)
        self.glpool = nn.AvgPool2d(kernel_size=32)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.ip = nn.Linear(1*1*128, 2)
        self.reset_parameters()

    def forward(self, x):
        x = x.float()
        x = self.pool(self.tanh(self.norm1(torch.abs(self.conv1(x)))))
        x = self.pool(self.tanh(self.norm2(self.conv2(x))))
        x = self.pool(self.relu(self.norm3(self.conv3(x))))
        x = self.pool(self.relu(self.norm4(self.conv4(x))))
        x = self.glpool(self.relu(self.norm5(self.conv5(x))))
        x = x.view(x.size(0), -1)
        x = self.ip(x)
        return x

    def reset_parameters(self):
        for mod in self.modules():
            if isinstance(mod, nn.Conv2d):
                nn.init.normal_(mod.weight, 0, 0.01)
            elif isinstance(mod, nn.BatchNorm2d):
                mod.reset_parameters()
            elif isinstance(mod, nn.Linear):
                nn.init.xavier_uniform(mod.weight)


def accuracy(outputs, labels):
    _, argmax = torch.max(outputs, 1)
    return (labels == argmax.squeeze()).float().mean()
