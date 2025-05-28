import os, sys
import torch
import torch.nn as nn
from ocr.modeling.backbones.rec_svtrnet import Block, ConvBNLayer

# 图像转为序列
class Im2Seq(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.out_channels = in_channels

    def forward(self, x):
        # 获取特征
        B, C, H, W = x.shape
        assert H == 1
        # 对h降维
        x = x.squeeze(dim=2)
        # 转换维度
        x = x.permute(0, 2, 1)
        return x


class EncoderWithRNN(nn.Module):
    def __init__(self, in_channels, hidden_size):
        super(EncoderWithRNN, self).__init__()
        self.out_channels = hidden_size * 2
        self.lstm = nn.LSTM(
            in_channels, hidden_size, num_layers=2, batch_first=True, bidirectional=True) # batch_first:=True

    def forward(self, x):
        x, _ = self.lstm(x)
        return x


class EncoderWithFC(nn.Module):
    def __init__(self, in_channels, hidden_size):
        super(EncoderWithFC, self).__init__()
        self.out_channels = hidden_size
        self.fc = nn.Linear(
            in_channels,
            hidden_size,
            bias=True,
            )

    def forward(self, x):
        x = self.fc(x)
        return x

# 基于空间变换器的编码
class EncoderWithSVTR(nn.Module):
    def __init__(
            self,
            in_channels,
            dims=64,  
            depth=2, # svtr堆叠层数
            hidden_dims=120, # svtr隐藏层维度
            use_guide=False,
            num_heads=8,
            qkv_bias=True, 
            mlp_ratio=2.0, # MLP层的扩展比率
            drop_rate=0.1,
            attn_drop_rate=0.1,
            drop_path=0.,
            kernel_size=[3, 3],
            qk_scale=None):
        super(EncoderWithSVTR, self).__init__()

        self.depth = depth
        self.use_guide = use_guide
        # 卷积
        self.conv1 = ConvBNLayer(
            in_channels,
            in_channels // 8,
            kernel_size=kernel_size,
            padding=[kernel_size[0] // 2, kernel_size[1] // 2],
            act='swish')
        # 1x1卷积
        self.conv2 = ConvBNLayer(in_channels // 8, hidden_dims, kernel_size=1, act='swish')
        # svtr 块
        self.svtr_block = nn.ModuleList([
            Block(
                dim=hidden_dims,
                num_heads=num_heads,
                mixer='Global',
                HW=None,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                act_layer='swish',
                attn_drop=attn_drop_rate,
                drop_path=drop_path,
                norm_layer='nn.LayerNorm',
                epsilon=1e-05,
                prenorm=False) for i in range(depth)
        ])
        # 归一化层
        self.norm = nn.LayerNorm(hidden_dims, eps=1e-6)
        self.conv3 = ConvBNLayer(hidden_dims, in_channels, kernel_size=1, act='swish')
        self.conv4 = ConvBNLayer(
            2 * in_channels,
            in_channels // 8,
            kernel_size=kernel_size,
            padding=[kernel_size[0] // 2, kernel_size[1] // 2],
            act='swish')

        self.conv1x1 = ConvBNLayer(in_channels // 8, dims, kernel_size=1, act='swish')
        self.out_channels = dims
        # 初始化参数
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # 截断正态分布
            nn.init.trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


    def forward(self, x):
        if self.use_guide:
            z = x.clone()
            z.stop_gradient = True
        else:
            z = x
        h = z
        z = self.conv1(z)
        z = self.conv2(z)
        B, C, H, W = z.shape
        z = z.flatten(2).permute([0, 2, 1])
        for blk in self.svtr_block:
            z = blk(z)
        z = self.norm(z)
        z = z.reshape([-1, H, W, C]).permute([0, 3, 1, 2])
        z = self.conv3(z)
        z = torch.cat((h, z), dim=1)
        z = self.conv1x1(self.conv4(z))
        return z

# 序列编码器
class SequenceEncoder(nn.Module):
    def __init__(self, in_channels, encoder_type, hidden_size=48, **kwargs):
        super(SequenceEncoder, self).__init__()
        self.encoder_reshape = Im2Seq(in_channels)
        self.out_channels = self.encoder_reshape.out_channels
        self.encoder_type = encoder_type
        
        support_encoder_dict = {
            'rnn': EncoderWithRNN,
            'svtr': EncoderWithSVTR,
        }
        assert encoder_type in support_encoder_dict, 'rec Neck层错误'

        if encoder_type == "svtr":
            self.encoder = support_encoder_dict[encoder_type](self.encoder_reshape.out_channels, **kwargs)
        else:
            self.encoder = support_encoder_dict[encoder_type](self.encoder_reshape.out_channels, hidden_size)
            
        self.out_channels = self.encoder.out_channels
        self.only_reshape = False

    def forward(self, x):
        if self.encoder_type != 'svtr':
            x = self.encoder_reshape(x)
            if not self.only_reshape:
                x = self.encoder(x)
            return x
        else:
            x = self.encoder(x)
            x = self.encoder_reshape(x)
            return x