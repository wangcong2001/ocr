import torch
import torch.nn as nn
import torch.nn.functional as F
from ocr.modeling.common import Activation
from ocr.modeling.backbones.det_mobilenet_v3 import SEModule

# class DSConv(nn.Module):
#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  kernel_size,
#                  padding,
#                  stride=1,
#                  groups=None,
#                  act="relu",
#                  **kwargs):
#         super(DSConv, self).__init__()
#         if groups == None:
#             groups = in_channels
#         self.act = act
#         self.conv1 = nn.Conv2d(
#             in_channels=in_channels,
#             out_channels=in_channels,
#             kernel_size=kernel_size,
#             stride=stride,
#             padding=padding,
#             groups=groups,
#             bias=False)

#         self.bn1 = nn.BatchNorm2d(in_channels)

#         self.conv2 = nn.Conv2d(
#             in_channels=in_channels,
#             out_channels=int(in_channels * 4),
#             kernel_size=1,
#             stride=1,
#             bias=False)

#         self.bn2 = nn.BatchNorm2d(int(in_channels * 4))

#         self.conv3 = nn.Conv2d(
#             in_channels=int(in_channels * 4),
#             out_channels=out_channels,
#             kernel_size=1,
#             stride=1,
#             bias=False)
#         self._c = [in_channels, out_channels]
#         if in_channels != out_channels:
#             self.conv_end = nn.Conv2d(
#                 in_channels=in_channels,
#                 out_channels=out_channels,
#                 kernel_size=1,
#                 stride=1,
#                 bias=False)
#         self.act = None
#         if act:
#             self.act = Activation(act)
#     def forward(self, inputs):

#         x = self.conv1(inputs)
#         x = self.bn1(x)

#         x = self.conv2(x)
#         x = self.bn2(x)
#         if self.act:
#             x = self.act(x)

#         x = self.conv3(x)
#         if self._c[0] != self._c[1]:
#             x = x + self.conv_end(inputs)
#         return x


# class DBFPN(nn.Module):
#     def __init__(self, in_channels, out_channels, use_asf=False, **kwargs):
#         super(DBFPN, self).__init__()
#         self.out_channels = out_channels
#         self.use_asf = use_asf

#         self.in2_conv = nn.Conv2d(
#             in_channels=in_channels[0],
#             out_channels=self.out_channels,
#             kernel_size=1,
#             bias=False)
#         self.in3_conv = nn.Conv2d(
#             in_channels=in_channels[1],
#             out_channels=self.out_channels,
#             kernel_size=1,
#             bias=False)
#         self.in4_conv = nn.Conv2d(
#             in_channels=in_channels[2],
#             out_channels=self.out_channels,
#             kernel_size=1,
#             bias=False)
#         self.in5_conv = nn.Conv2d(
#             in_channels=in_channels[3],
#             out_channels=self.out_channels,
#             kernel_size=1,
#             bias=False)
#         self.p5_conv = nn.Conv2d(
#             in_channels=self.out_channels,
#             out_channels=self.out_channels // 4,
#             kernel_size=3,
#             padding=1,
#             bias=False)
#         self.p4_conv = nn.Conv2d(
#             in_channels=self.out_channels,
#             out_channels=self.out_channels // 4,
#             kernel_size=3,
#             padding=1,
#             bias=False)
#         self.p3_conv = nn.Conv2d(
#             in_channels=self.out_channels,
#             out_channels=self.out_channels // 4,
#             kernel_size=3,
#             padding=1,
#             bias=False)
#         self.p2_conv = nn.Conv2d(
#             in_channels=self.out_channels,
#             out_channels=self.out_channels // 4,
#             kernel_size=3,
#             padding=1,
#             bias=False)

#         if self.use_asf is True:
#             self.asf = ASFBlock(self.out_channels, self.out_channels // 4)

#     def forward(self, x):
#         c2, c3, c4, c5 = x

#         in5 = self.in5_conv(c5)
#         in4 = self.in4_conv(c4)
#         in3 = self.in3_conv(c3)
#         in2 = self.in2_conv(c2)

#         out4 = in4 + F.interpolate(
#             in5, scale_factor=2, mode="nearest", )#align_mode=1)  # 1/16
#         out3 = in3 + F.interpolate(
#             out4, scale_factor=2, mode="nearest", )#align_mode=1)  # 1/8
#         out2 = in2 + F.interpolate(
#             out3, scale_factor=2, mode="nearest", )#align_mode=1)  # 1/4

#         p5 = self.p5_conv(in5)
#         p4 = self.p4_conv(out4)
#         p3 = self.p3_conv(out3)
#         p2 = self.p2_conv(out2)
#         p5 = F.interpolate(p5, scale_factor=8, mode="nearest", )#align_mode=1)
#         p4 = F.interpolate(p4, scale_factor=4, mode="nearest", )#align_mode=1)
#         p3 = F.interpolate(p3, scale_factor=2, mode="nearest", )#align_mode=1)

#         fuse = torch.cat([p5, p4, p3, p2], dim=1)

#         if self.use_asf is True:
#             fuse = self.asf(fuse, [p5, p4, p3, p2])

#         return fuse

# 
class RSELayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, shortcut=True):
        super(RSELayer, self).__init__()
        self.out_channels = out_channels
        # 二维卷积
        self.in_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.out_channels,
            kernel_size=kernel_size,
            padding=int(kernel_size // 2),
            bias=False)
        # SE模块
        self.se_block = SEModule(self.out_channels)
        # 残差块链接
        self.shortcut = shortcut

    def forward(self, ins):
        # 通过卷积层
        x = self.in_conv(ins)
        # 通过SE模块
        if self.shortcut:
            out = x + self.se_block(x)
        else:
            out = self.se_block(x)
        return out


class RSEFPN(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True, **kwargs):
        super(RSEFPN, self).__init__()
        # 获取输出通道数
        self.out_channels = out_channels
        # 定义输入卷积层
        self.ins_conv = nn.ModuleList()
        # 定义输出卷积层
        self.inp_conv = nn.ModuleList()

        for i in range(len(in_channels)):
            # 为每个通道创建RSELayer
            self.ins_conv.append(
                RSELayer(
                    in_channels[i],
                    out_channels,
                    kernel_size=1,
                    shortcut=shortcut))
            self.inp_conv.append(
                RSELayer(
                    out_channels,
                    out_channels // 4,
                    kernel_size=3,
                    shortcut=shortcut))

    def forward(self, x):
        # 获取四个变量
        c2, c3, c4, c5 = x
        # 通过卷积层
        in5 = self.ins_conv[3](c5)
        in4 = self.ins_conv[2](c4)
        in3 = self.ins_conv[1](c3)
        in2 = self.ins_conv[0](c2)
        # 上采样 特征融合
        out4 = in4 + F.upsample(in5, scale_factor=2, mode="nearest")  # 1/16
        out3 = in3 + F.upsample(out4, scale_factor=2, mode="nearest")  # 1/8
        out2 = in2 + F.upsample(out3, scale_factor=2, mode="nearest")  # 1/4
        # 再次进入卷积层
        p5 = self.inp_conv[3](in5)
        p4 = self.inp_conv[2](out4)
        p3 = self.inp_conv[1](out3)
        p2 = self.inp_conv[0](out2)
        # 上采样 并调整空间尺寸
        p5 = F.upsample(p5, scale_factor=8, mode="nearest")
        p4 = F.upsample(p4, scale_factor=4, mode="nearest")
        p3 = F.upsample(p3, scale_factor=2, mode="nearest")
        # 拼接操作
        fuse = torch.cat([p5, p4, p3, p2], dim=1)
        return fuse

