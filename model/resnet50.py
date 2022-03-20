# -*- coding: utf-8 -*-
import torch
from torch import nn

class resnet50(nn.Module):
    def __init__(self):
        super(resnet50, self).__init__()

    def forward(self, x):
        pass
    

class BottleNeck_Block(nn.Module):
    def __init__(self, layer=None, downsample=True):
        super(BottleNeck_Block, self).__init__()
        assert isinstance(layer, int)
        self.layer = layer
        self.conv1 = nn.Conv2d(self.layer**2*64, (self.layer-1)**2*64,
                               kernel_size=(1, 1), stride=(1, 1))
        self.actv1 = nn.ReLU(inplace=True)
        if downsample:
            self.conv2 = nn.Conv2d(self.layer**2*64, (self.layer-1)**2*64,
                                   kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.conv4 = nn.Conv2d(self.layer**2*64, (self.layer+1)**2*64,
                                   kernel_size=(1, 1), stride=(1, 1))
        else:
            self.conv2 = nn.Conv2d(self.layer ** 2 * 64, (self.layer - 1) ** 2 * 64,
                                   kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.conv4 = nn.Conv2d(self.layer ** 2 * 64, (self.layer + 1) ** 2 * 64,
                                   kernel_size=(1, 1), stride=(1, 1))
        self.conv3 = nn.Conv2d((self.layer-1)**2*64, (self.layer+1)**2*64,
                               kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        pass
