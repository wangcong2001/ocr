from torch import nn

from .rec_ctc_loss import CTCLoss
from .rec_sar_loss import SARLoss


class MultiLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.loss_funcs = {}
        self.loss_list = kwargs.pop('loss_config_list')
        self.weight_1 = kwargs.get('weight_1', 1.0)
        self.weight_2 = kwargs.get('weight_2', 1.0)

        for loss_info in self.loss_list:
            for name, param in loss_info.items():
                if param is not None:
                    kwargs.update(param)
                loss = eval(name)(**kwargs)
                self.loss_funcs[name] = loss

    def forward(self, predicts, batch):
        self.total_loss = {}
        total_loss = 0.0
        for name, loss_func in self.loss_funcs.items():
            if name == 'CTCLoss':
                loss = loss_func({'res': predicts['ctc']}, batch[:2] + batch[3:])['loss'] * self.weight_1
            elif name == 'SARLoss':
                loss = loss_func({'res': predicts['sar']}, batch[:1] + batch[2:])['loss'] * self.weight_2
            else:
                raise NotImplementedError('rec 损失函数错误')
            self.total_loss[name] = loss
            total_loss += loss
        # 返货总损失
        self.total_loss['loss'] = total_loss
        return self.total_loss
