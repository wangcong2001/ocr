import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from OCR.pytorchocr.modeling.common import Activation


class Head(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super(Head, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels // 4,
            kernel_size=3,
            padding=1,
            bias=False)
        self.conv_bn1 = nn.BatchNorm2d(
            in_channels // 4)
        self.relu1 = Activation(act_type='relu')

        self.conv2 = nn.ConvTranspose2d(
            in_channels=in_channels // 4,
            out_channels=in_channels // 4,
            kernel_size=2,
            stride=2)
        self.conv_bn2 = nn.BatchNorm2d(
            in_channels // 4)
        self.relu2 = Activation(act_type='relu')

        self.conv3 = nn.ConvTranspose2d(
            in_channels=in_channels // 4,
            out_channels=1,
            kernel_size=2,
            stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv_bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.conv_bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = torch.sigmoid(x)
        return x


class DBHead(nn.Module):


    def __init__(self, in_channels, k=50, **kwargs):
        super(DBHead, self).__init__()
        self.k = k
        binarize_name_list = [
            'conv2d_56', 'batch_norm_47', 'conv2d_transpose_0', 'batch_norm_48',
            'conv2d_transpose_1', 'binarize'
        ]
        thresh_name_list = [
            'conv2d_57', 'batch_norm_49', 'conv2d_transpose_2', 'batch_norm_50',
            'conv2d_transpose_3', 'thresh'
        ]
        self.binarize = Head(in_channels,)# binarize_name_list)
        self.thresh = Head(in_channels, )#thresh_name_list)

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))

    def forward(self, x):
        shrink_maps = self.binarize(x)
        if not self.training:
            return {'maps': shrink_maps}

        threshold_maps = self.thresh(x)
        binary_maps = self.step_function(shrink_maps, threshold_maps)
        y = torch.cat([shrink_maps, threshold_maps, binary_maps], dim=1)
        return {'maps': y}