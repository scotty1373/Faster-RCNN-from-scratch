# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from torchvision.models import resnet50


class Resnet50(nn.Module):
    def __init__(self, block, layers, num_class):
        super(Resnet50, self).__init__()
        self.block = block
        self.layers = layers

        self.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        self.layer1 = self.get_block(64, 64, self.layers[0])
        self.layer2 = self.get_block(256, 128, self.layers[1], strides=(2, 2))
        self.layer3 = self.get_block(512, 256, self.layers[2], strides=(2, 2))
        self.layer4 = self.get_block(1024, 512, self.layers[3], strides=(2, 2))

        self.global_pooling = nn.AvgPool2d(kernel_size=(7, 7))
        self.fc = nn.Linear(2048, num_class)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.global_pooling(out)
        out = torch.flatten(out, start_dim=1, end_dim=-1)
        out = self.fc(out)
        return out

    # 确保残差块初始化时strides正确
    def get_block(self, channel_in, channel_neck, block_repeat_num, strides=(1, 1)):
        block = []
        for idx in range(block_repeat_num):
            if not idx:
                block.append(BottleNeck_Block(channel_in, channel_neck, strides))
            else:
                block.append(BottleNeck_Block(channel_neck * self.block.extention, channel_neck))
        return nn.Sequential(*block)
    

class BottleNeck_Block(nn.Module):

    extention = 4

    # 增加strides保证Identity Block结构步长为1
    def __init__(self, channel_in, channel_neck, strides=(1, 1)):
        """

        :param channel_in: 输入维度
        :param channel_neck: 中间neck维度，通常为输出维度的1/4
        :param strides: 降采样步长
        """
        super(BottleNeck_Block, self).__init__()

        self.conv1 = nn.Conv2d(channel_in, channel_neck,
                               kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(channel_neck)
        self.act1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(channel_neck, channel_neck,
                               kernel_size=(3, 3), stride=strides, padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(channel_neck)
        self.act2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(channel_neck, self.extention*channel_neck,
                               kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(self.extention*channel_neck)

        self.downsample = None
        if strides != (1, 1) or channel_in != self.extention*channel_neck:
            downsample = [nn.Conv2d(channel_in, self.extention*channel_neck,
                                    kernel_size=(1, 1), stride=strides, bias=False),
                          nn.BatchNorm2d(self.extention*channel_neck)]
            self.downsample = nn.Sequential(*downsample)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.act1(out)
        out = self.bn2(self.conv2(out))
        out = self.act2(out)
        out = self.bn3(self.conv3(out))
        # Down sample opt
        if self.downsample is not None:
            out += self.downsample(x)
        else:
            out += x
        return F.relu(out)


if __name__ == '__main__':
    pretrained_path = '/Users/scotty/Downloads/resnet50-19c8e357.pth'
    ckpt = torch.load(pretrained_path)
    x = torch.randn(10, 3, 224, 224)
    model = Resnet50(BottleNeck_Block, [3, 4, 6, 3], 1000)
    model_pre = resnet50(pretrained=False)
    model_pre.load_state_dict(ckpt)

    model.load_state_dict(ckpt)

    model(x)
    print(model)




