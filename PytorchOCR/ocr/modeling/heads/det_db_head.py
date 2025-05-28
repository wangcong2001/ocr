import torch
from torch import nn
import torch.nn.functional as F
from ocr.modeling.backbones.det_mobilenet_v3 import ConvBNLayer



class Head(nn.Module):
    def __init__(self, in_channels, kernel_list=[3, 2, 2], **kwargs):
        super(Head, self).__init__()
        # 卷积
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels // 4,
            kernel_size=kernel_list[0],
            padding=int(kernel_list[0] // 2),
            bias=False)
        # BN
        self.conv_bn1 = nn.BatchNorm2d(in_channels // 4)
        # 转置卷积 2倍上采样
        self.conv2 = nn.ConvTranspose2d(
            in_channels=in_channels // 4,
            out_channels=in_channels // 4,
            kernel_size=kernel_list[1],
            stride=2)
        # BN
        self.conv_bn2 = nn.BatchNorm2d(in_channels // 4)
        # 转置卷积  2倍上采样
        self.conv3 = nn.ConvTranspose2d(
            in_channels=in_channels // 4,
            out_channels=1,
            kernel_size=kernel_list[2],
            stride=2)

    def forward(self, x, return_f=False):
        x = self.conv1(x)
        x = self.conv_bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.conv_bn2(x)
        x = F.relu(x)
        # 是否返回中间特征图f
        if return_f is True:
            f = x
        x = self.conv3(x)
        x = F.sigmoid(x)
        if return_f is True:
            return x, f
        return x


class DBHead(nn.Module):
    def __init__(self, in_channels, k=50, **kwargs):
        super(DBHead, self).__init__()
        # 获取参数k
        self.k = k
        # 定义收缩和阈值图
        self.binarize = Head(in_channels, **kwargs)
        self.thresh = Head(in_channels, **kwargs)
    # 定义阶跃函数
    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))

    def forward(self, x, data=None):
        # 获取收缩图
        shrink_maps = self.binarize(x)
        # 非训练模式
        if not self.training:
            # 仅返回收缩图
            return {'res': shrink_maps}
        # 获取阈值图
        threshold_maps = self.thresh(x)
        # 获取二值图
        binary_maps = self.step_function(shrink_maps, threshold_maps)
        # 拼接
        y = torch.cat([shrink_maps, threshold_maps, binary_maps], dim=1)
        return {'res': y}
