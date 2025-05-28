import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

import copy
from torch.utils.data import DataLoader, DistributedSampler

from ocr.data.imaug import transform, create_operators
from ocr.data.simple_dataset import SimpleDataSet

__all__ = [
    'build_dataloader', 'transform', 'create_operators',
]


def build_dataloader(config, mode, logger, seed=None):
    # 复制配置文件
    config = copy.deepcopy(config)
    # 获取数据集名称
    module_name = config[mode]['dataset']['name']
    # 获取当前模式
    assert mode in ['Train', 'Eval', 'Test'], "当前模式错误"
    # 实例化数据集
    dataset = eval(module_name)(config, mode, logger, seed)
    # 获取加载器配置
    loader_config = config[mode]['loader']
    # 获取批次大小
    batch_size = loader_config['batch_size_per_card']
    # 是否丢弃最后一个批次
    drop_last = loader_config['drop_last']
    # 是否打乱数据
    shuffle = loader_config['shuffle']
    # 获取工作进程数
    num_workers = loader_config['num_workers']
    # 是否加载进固定内存
    if 'pin_memory' in loader_config.keys():
        pin_memory = loader_config['use_shared_memory']
    else:
        pin_memory = False
    # 初始化采样器
    sampler = None
    batch_sampler=None

    if mode == "Train":
        # 指定采样器
        if 'sampler' in config[mode]:
            # 获取参数
            config_sampler = config[mode]['sampler']
            # 获取采样器名称
            sampler_name = config_sampler.pop("name")
            # 实例化采样器
            batch_sampler = eval(sampler_name)(dataset, **config_sampler)
        # 是否分布式
        elif config['Global']['distributed']:
            # 实例化分布式采样器
            sampler = DistributedSampler(dataset=dataset, shuffle=shuffle)

    # 是否指定拼接函数
    if 'collate_fn' in loader_config:
        # 导入
        from . import collate_fn
        # 实例化
        collate_fn = getattr(collate_fn, loader_config['collate_fn'])()
    else:
        collate_fn = None
    # 没有指定采样器
    if batch_sampler is None:
        # 实例化加载器
        data_loader = DataLoader(
            dataset=dataset,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            batch_size=batch_size,
            drop_last=drop_last
        )
    else:
        # 实例化加载器
        data_loader = DataLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )
    if len(data_loader) == 0:
        logger.error("Dataloader 错误\n")
        sys.exit()
    return data_loader
