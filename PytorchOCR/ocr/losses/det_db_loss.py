import torch
from torch import nn

from .det_basic_loss import BalanceLoss, MaskL1Loss, DiceLoss

# 损失函数
class DBLoss(nn.Module):
    def __init__(self, 
                 balance_loss=True, # 
                 main_loss_type='DiceLoss',  # 主要的损失函数类型
                 alpha=5, # 调节收缩图损失的系数
                 beta=10, # 调节二值图损失的系数
                 ohem_ratio=3, # OHEM 比例
                 eps=1e-6,
                 **kwargs):
        super(DBLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.dice_loss = DiceLoss(eps=eps)
        self.l1_loss = MaskL1Loss(eps=eps)
        self.bce_loss = BalanceLoss(
            balance_loss=balance_loss,
            main_loss_type=main_loss_type,
            negative_ratio=ohem_ratio)

    def forward(self, predicts, labels):
        predict_maps = predicts['res']
        label_threshold_map, label_threshold_mask, label_shrink_map, label_shrink_mask = labels[1:]
        shrink_maps = predict_maps[:, 0, :, :]
        threshold_maps = predict_maps[:, 1, :, :]
        binary_maps = predict_maps[:, 2, :, :]
        # 使用 BalanceLoss 计算收缩图损失 loss_shrink_maps
        loss_shrink_maps = self.bce_loss(shrink_maps, label_shrink_map,label_shrink_mask)
        # 使用 MaskL1Loss 计算阈值图损失 loss_threshold_maps
        loss_threshold_maps = self.l1_loss(threshold_maps, label_threshold_map,label_threshold_mask)
        # 使用 DiceLoss 计算二值图损失 loss_binary_maps
        loss_binary_maps = self.dice_loss(binary_maps, label_shrink_map, label_shrink_mask)
        # 对收缩图损失和阈值图损失进行加权
        loss_shrink_maps = self.alpha * loss_shrink_maps
        loss_threshold_maps = self.beta * loss_threshold_maps
        # 预测结果中存在距离图和 CBN 图计算
        if 'distance_maps' in predicts.keys():
            distance_maps = predicts['distance_maps']
            cbn_maps = predicts['cbn_maps']
            cbn_loss = self.bce_loss(cbn_maps[:, 0, :, :], label_shrink_map, label_shrink_mask)
        else:
            dis_loss = torch.tensor([0.], device=shrink_maps.device)
            cbn_loss = torch.tensor([0.], device=shrink_maps.device)
        

        loss_all = loss_shrink_maps + loss_threshold_maps + loss_binary_maps
        losses = {'loss': loss_all+ cbn_loss, \
                  "loss_shrink_maps": loss_shrink_maps, \
                  "loss_threshold_maps": loss_threshold_maps, \
                  "loss_binary_maps": loss_binary_maps, \
                  "loss_cbn": cbn_loss}
        return losses
