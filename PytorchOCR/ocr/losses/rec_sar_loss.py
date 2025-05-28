from torch import nn

# SAR loss
class SARLoss(nn.Module):
    def __init__(self, **kwargs):
        super(SARLoss, self).__init__()
        ignore_index = kwargs.get('ignore_index', 92) 
        # 使用CrossEntropyLoss
        self.loss_func = nn.CrossEntropyLoss(reduction="mean", ignore_index=ignore_index)

    def forward(self, predicts, batch):
        predicts = predicts['res']
        predict = predicts[:, :-1, :]  
        label = batch[1].long()[:, 1:] 
        batch_size, num_steps, num_classes = predict.shape[0], predict.shape[1], predict.shape[2]
        assert len(label.shape) == len(list(predict.shape)) - 1
        # 展成二维张量
        inputs = predict.reshape([-1, num_classes])
        targets = label.reshape([-1])
        loss = self.loss_func(inputs, targets)
        return {'loss': loss}
