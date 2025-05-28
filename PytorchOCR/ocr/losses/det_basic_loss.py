import torch
from torch import nn
import torch.nn.functional as F

# 平衡损失
class BalanceLoss(nn.Module):
    def __init__(self,
                 balance_loss=True,
                 main_loss_type='DiceLoss',
                 negative_ratio=3, # 负样本比例
                 return_origin=False, # 是否返回原始损失
                 eps=1e-6,
                 **kwargs):
        super(BalanceLoss, self).__init__()
        self.balance_loss = balance_loss
        self.main_loss_type = main_loss_type
        self.negative_ratio = negative_ratio
        self.return_origin = return_origin
        self.eps = eps
        # 构建损失
        if self.main_loss_type == "CrossEntropy":
            self.loss = nn.CrossEntropyLoss()
        elif self.main_loss_type == "Euclidean":
            self.loss = nn.MSELoss()
        elif self.main_loss_type == "DiceLoss":
            self.loss = DiceLoss(self.eps)
        elif self.main_loss_type == "MaskL1Loss":
            self.loss = MaskL1Loss(self.eps)
        else:
            raise Exception('det 损失函数错误')

    def forward(self, pred, gt, mask=None):
        positive = gt * mask
        negative = (1 - gt) * mask
        # 正样本数量
        positive_count = int(positive.sum())
        # 负样本数量
        negative_count = int(min(negative.sum(), positive_count * self.negative_ratio))
        # 计算损失
        loss = self.loss(pred, gt, mask=mask)

        if not self.balance_loss:
            return loss
        # 正负损失
        positive_loss = positive * loss
        negative_loss = negative * loss
        # 对负样本重塑
        negative_loss = torch.reshape(negative_loss, shape=[-1])
        if negative_count > 0:
            # 排序 
            sort_loss, _ = negative_loss.sort(descending=True)
            negative_loss = sort_loss[:negative_count]
            # 计算平衡损失
            balance_loss = (positive_loss.sum() + negative_loss.sum()) / (positive_count + negative_count + self.eps)
        else:
            balance_loss = positive_loss.sum() / (positive_count + self.eps)
        if self.return_origin:
            return balance_loss, loss

        return balance_loss


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(DiceLoss, self).__init__()
        self.eps = eps
    # 预测 标签 掩码 权重
    def forward(self, pred, gt, mask, weights=None):
        assert pred.shape == gt.shape
        assert pred.shape == mask.shape
        if weights is not None:
            assert weights.shape == mask.shape
            mask = weights * mask
        # 计算三者交集
        intersection = torch.sum(pred * gt * mask)
        # 计算并集
        union = torch.sum(pred * mask) + torch.sum(gt * mask) + self.eps
        # 计算损失
        loss = 1 - 2.0 * intersection / union
        assert loss <= 1
        return loss


class MaskL1Loss(nn.Module):
    def __init__(self, eps=1e-6):
        super(MaskL1Loss, self).__init__()
        self.eps = eps
    # 预测 标签 掩码
    def forward(self, pred, gt, mask):
        # 计算损失
        loss = (torch.abs(pred - gt) * mask).sum() / (mask.sum() + self.eps)
        # 平均损失
        loss = torch.mean(loss)
        return loss