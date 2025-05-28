import os

import torch

from ocr.utils.logging import get_logger

# 保存检查点
def save_ckpt(model, cfg, optimizer, lr_scheduler, epoch, global_step, metrics, is_best=False, logger=None):
    if logger is None:
        logger = get_logger()
    # 是否是最佳模型
    if is_best:
        save_path = os.path.join(cfg['Global']['output_dir'], 'best.pth')
    else:
        save_path = os.path.join(cfg['Global']['output_dir'], 'latest.pth')
    # 模型参数
    state_dict = model.module.state_dict() if cfg['Global']['distributed'] else model.state_dict()
    # 保存状态
    state = {
        'epoch': epoch,
        'global_step': global_step,
        'state_dict': state_dict,
        'optimizer': optimizer.state_dict(),
        'scheduler': lr_scheduler.state_dict(),
        'config': cfg,
        'metrics': metrics
    }
    torch.save(state, save_path)
    logger.info(f'保存模型参数 {save_path}')

# 加载检查点回复训练状态
def load_ckpt(model, cfg, optimizer=None, lr_scheduler=None, logger=None):
    if logger is None:
        logger = get_logger()
    # 获取检查点
    checkpoints = cfg['Global'].get('checkpoints')
    # 获取预训练模型
    pretrained_model = cfg['Global'].get('pretrained_model')

    status = {}
    # 存在路径
    if checkpoints and os.path.exists(checkpoints):
        # 加载检查点
        checkpoint = torch.load(checkpoints, map_location=torch.device('cpu'))
        # 加载模型参数
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        # 加载优化器参数
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        # 加载学习率调度器参数
        if lr_scheduler is not None:
            lr_scheduler.load_state_dict(checkpoint['scheduler'])
        # 打印日志
        logger.info(f"从检查点恢复 {checkpoints} (epoch {checkpoint['epoch']})")
        # 更新状态
        status['global_step'] = checkpoint['global_step']
        status['epoch'] = checkpoint['epoch'] + 1
        status['metrics'] = checkpoint['metrics']
    # 存在预训练模型
    elif pretrained_model and os.path.exists(pretrained_model):
        # 加载预训练模型参数
        load_pretrained_params(model, pretrained_model)
        logger.info(f"从检查点微调{pretrained_model}")
    else:
        logger.info("训练开始")
    return status


def load_pretrained_params(model, pretrained_model):
    # 加载预训练模型参数
    checkpoint = torch.load(pretrained_model, map_location=torch.device('cpu'))
    # 加载模型参数
    model.load_state_dict(checkpoint['state_dict'], strict=False)
