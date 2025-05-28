import torch.nn as nn
from ocr.modeling.common import Activation


# 进行通道调整
def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# 卷积 BN层
class ConvBNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 groups=1,
                 if_act=True,
                 act=None):

        super(ConvBNLayer, self).__init__()
        self.if_act = if_act
        # 二维卷积
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False)
        # 二维批量归一化
        self.bn = nn.BatchNorm2d(out_channels, )
        if self.if_act:
            self.act = Activation(act_type=act, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.if_act:
            x = self.act(x)
        return x


class SEModule(nn.Module):
    def __init__(self, in_channels, reduction=4, name=""):
        super(SEModule, self).__init__()
        # 平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 卷积
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels // reduction,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)
        # 激活函数
        self.relu1 = Activation(act_type='relu', inplace=True)
        # 卷积
        self.conv2 = nn.Conv2d(
            in_channels=in_channels // reduction,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)
        # 激活
        self.hard_sigmoid = Activation(act_type='hard_sigmoid', inplace=True)

    def forward(self, inputs):
        outputs = self.avg_pool(inputs)
        outputs = self.conv1(outputs)
        outputs = self.relu1(outputs)
        outputs = self.conv2(outputs)
        outputs = self.hard_sigmoid(outputs)
        outputs = inputs * outputs
        return outputs


# 残差块
class ResidualUnit(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 use_se,
                 act=None,
                 name=''):
        super(ResidualUnit, self).__init__()

        self.if_shortcut = stride == 1 and in_channels == out_channels
        self.if_se = use_se
        # 扩张卷积
        self.expand_conv = ConvBNLayer(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=1,  # 1X1卷积核
            stride=1,
            padding=0,
            if_act=True,
            act=act)
        # 瓶颈卷积 特征提取
        self.bottleneck_conv = ConvBNLayer(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,  # 配置卷积核
            stride=stride,
            padding=int((kernel_size - 1) // 2),  # 填充大小
            groups=mid_channels,  # 分组卷积
            if_act=True,
            act=act)

        # 添加se模块
        if self.if_se:
            self.mid_se = SEModule(mid_channels, name=name + "_se")
        # 线性变换
        self.linear_conv = ConvBNLayer(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            if_act=False,
            act=None)

    def forward(self, inputs):
        x = self.expand_conv(inputs)
        x = self.bottleneck_conv(x)
        if self.if_se:
            x = self.mid_se(x)
        x = self.linear_conv(x)
        if self.if_shortcut:
            x = inputs + x
        return x


class MobileNetV3(nn.Module):
    def __init__(self,
                 in_channels=3,
                 model_name='large',
                 scale=0.5,
                 disable_se=False,
                 **kwargs):
        super(MobileNetV3, self).__init__()
        # 是否使用se模块
        self.disable_se = disable_se

        if model_name == "large":
            cfg = [
                # 卷积核 扩张系数 输出通道数 se模块 激活函数 步长
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, False, 'relu', 1],
                [3, 64, 24, False, 'relu', 2],
                [3, 72, 24, False, 'relu', 1],
                [5, 72, 40, True, 'relu', 2],
                [5, 120, 40, True, 'relu', 1],
                [5, 120, 40, True, 'relu', 1],
                [3, 240, 80, False, 'hard_swish', 2],
                [3, 200, 80, False, 'hard_swish', 1],
                [3, 184, 80, False, 'hard_swish', 1],
                [3, 184, 80, False, 'hard_swish', 1],
                [3, 480, 112, True, 'hard_swish', 1],
                [3, 672, 112, True, 'hard_swish', 1],
                [5, 672, 160, True, 'hard_swish', 2],
                [5, 960, 160, True, 'hard_swish', 1],
                [5, 960, 160, True, 'hard_swish', 1],
            ]
            cls_ch_squeeze = 960
        elif model_name == "small":
            cfg = [
                [3, 16, 16, True, 'relu', 2],
                [3, 72, 24, False, 'relu', 2],
                [3, 88, 24, False, 'relu', 1],
                [5, 96, 40, True, 'hard_swish', 2],
                [5, 240, 40, True, 'hard_swish', 1],
                [5, 240, 40, True, 'hard_swish', 1],
                [5, 120, 48, True, 'hard_swish', 1],
                [5, 144, 48, True, 'hard_swish', 1],
                [5, 288, 96, True, 'hard_swish', 2],
                [5, 576, 96, True, 'hard_swish', 1],
                [5, 576, 96, True, 'hard_swish', 1],
            ]
            cls_ch_squeeze = 576
        else:
            raise NotImplementedError("主干网络错误")

        # 预定义比例
        supported_scale = [0.35, 0.5, 0.75, 1.0, 1.25]
        assert scale in supported_scale, "比例错误"
        # 输入的通道数
        inplanes = 16
        # 卷积 BN层
        self.conv = ConvBNLayer(
            in_channels=in_channels,
            out_channels=make_divisible(inplanes * scale),
            kernel_size=3,
            stride=2,
            padding=1,
            groups=1,
            if_act=True,
            act='hard_swish')
        # 模块列表
        self.stages = nn.ModuleList()
        # 存储输出通道
        self.out_channels = []
        # 存储块
        block_list = []
        i = 0
        # 输出通道数
        inplanes = make_divisible(inplanes * scale)
        # 遍历配置
        for (k, exp, c, se, nl, s) in cfg:
            # 是否使用se
            se = se and not self.disable_se
            # 下采样
            if s == 2 and i > 2:
                # 添加模块列表
                self.out_channels.append(inplanes)
                self.stages.append(nn.Sequential(*block_list))
                block_list = []
            # 残差块添加
            block_list.append(
                ResidualUnit(
                    in_channels=inplanes,
                    mid_channels=make_divisible(scale * exp),  # 中间的通道
                    out_channels=make_divisible(scale * c),  # 输出的通道
                    kernel_size=k,
                    stride=s,
                    use_se=se,
                    act=nl,
                    name="conv" + str(i + 2)))
            # 输入通道为上次的输出
            inplanes = make_divisible(scale * c)
            i += 1
        # 添加卷积 BN层
        block_list.append(
            ConvBNLayer(
                in_channels=inplanes,
                out_channels=make_divisible(scale * cls_ch_squeeze),
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                if_act=True,
                act='hard_swish'))
        # 添加模块列表
        self.stages.append(nn.Sequential(*block_list))
        # 输出通道
        self.out_channels.append(make_divisible(scale * cls_ch_squeeze))

    def forward(self, x):
        x = self.conv(x)
        # 输出列表
        out_list = []
        # 遍历模块列表
        for stage in self.stages:
            x = stage(x)
            out_list.append(x)
        return out_list
