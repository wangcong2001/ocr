import torch
import torch.nn as nn

from ocr.modeling.necks.rnn import Im2Seq, SequenceEncoder
from .rec_ctc_head import CTCHead
from .rec_sar_head import SARHead



class MultiHead(nn.Module):
    def __init__(self, in_channels, out_channels_list, **kwargs):
        super().__init__()
        # 获取Head列表
        self.head_list = kwargs.pop('head_list')
        self.gtc_head = 'sar'
        assert len(self.head_list) >= 2
        # 遍历head列表
        for idx, head_name in enumerate(self.head_list):
            name = list(head_name)[0]
            # SAR head
            if name == 'SARHead':
                sar_args = self.head_list[idx][name]
                self.sar_head = eval(name)(in_channels=in_channels, out_channels=out_channels_list['SARLabelDecode'], **sar_args)
            elif name == 'CTCHead':
                # ctc Neck层
                self.encoder_reshape = Im2Seq(in_channels)
                # 获取参数
                neck_args = self.head_list[idx][name]['Neck']
                encoder_type = neck_args.pop('name')
                # 序列编码器
                self.ctc_encoder = SequenceEncoder(in_channels=in_channels, encoder_type=encoder_type, **neck_args)
                # ctc Head层
                head_args = self.head_list[idx][name].get('Head', {})
                if head_args is None:
                    head_args = {}
                self.ctc_head = eval(name)(in_channels=self.ctc_encoder.out_channels, out_channels=out_channels_list['CTCLabelDecode'], **head_args)
            else:
                raise NotImplementedError('rec Head 错误')

    def forward(self, x, data=None):
        # 序列编码
        ctc_encoder = self.ctc_encoder(x)
        # ctc head
        ctc_out = self.ctc_head(ctc_encoder)['res']
        head_out = dict()

        head_out['ctc'] = ctc_out
        head_out['res'] = ctc_out
        head_out['ctc_neck'] = ctc_encoder

        # 模型评估
        if not self.training:
            return {'res': ctc_out}
        # 训练模式 SAR head
        if self.gtc_head == 'sar':
            sar_out = self.sar_head(x, data[1:])['res']
            head_out['sar'] = sar_out
        return head_out
