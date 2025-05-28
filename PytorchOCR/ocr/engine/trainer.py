import datetime
import os
import random
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from tools.utility import update_rec_head_out_channels
from ocr.data import build_dataloader
from ocr.losses import build_loss
from ocr.metrics import build_metric
from ocr.modeling.architectures import build_model
from ocr.optimizer import build_optimizer
from ocr.postprocess import build_post_process
from ocr.utils.ckpt import save_ckpt, load_ckpt
from ocr.utils.logging import get_logger
from ocr.utils.stats import TrainingStats
from ocr.utils.utility import AverageMeter

__all__ = ['Trainer']


class Trainer(object):
    def __init__(self, cfg, mode='train'):
        
        # 获取参数
        self.cfg = cfg.cfg
        # 获取本地GPUID
        self.local_rank = int(os.environ['LOCAL_RANK']) if 'LOCAL_RANK' in os.environ else 0
        # 是否使用GPU或CPU
        self.set_device(self.cfg['Global']['device'])
        # 是否使用多GPU
        if torch.cuda.device_count() > 1:
            # 初始化分布式训练环境，使用 NCCL 作为后端。
            torch.distributed.init_process_group(backend="nccl")
            torch.cuda.set_device(self.device)
            # 加入配置
            self.cfg['Global']['distributed'] = True
        else:
            # 单卡
            self.cfg['Global']['distributed'] = False
            self.local_rank = 0
        
        # 创建输出目录
        self.cfg['Global']['output_dir'] = self.cfg['Global'].get('output_dir', 'output')
        os.makedirs(self.cfg['Global']['output_dir'], exist_ok=True)

        # 是否创建 TensorBoard 的 SummaryWriter 对象
        self.writer = None
        if self.local_rank == 0 and self.cfg['Global']['use_tensorboard'] and 'train' in mode:
            self.writer = SummaryWriter(self.cfg['Global']['output_dir'])

        # 启动日志记录
        self.logger = get_logger('ocr', os.path.join(self.cfg['Global']['output_dir'],  'train.log') if 'train' in mode else None)
        
        # cfg.print_cfg(self.logger.info)

        # 检测cuda是否可用
        if self.cfg['Global']['device'] == 'gpu' and self.device.type == 'cpu':
            self.logger.info('CUDA不可用')

        # 设置随机种子
        self.set_random_seed(self.cfg['Global'].get('seed', 48))

        # 查看当前模式 训练 评估 测试
        mode = mode.lower()
        assert mode in ['train_eval', 'train', 'eval', 'test'], "当前模式错误"

        # 构建dataloader
        self.train_dataloader = None
        if 'train' in mode:
            # 保存当前配置
            cfg.save(os.path.join(self.cfg['Global']['output_dir'], 'config.yml'), self.cfg)
            # 构建训练数据集加载器
            self.train_dataloader = build_dataloader(self.cfg, 'Train', self.logger)
            # 添加日志
            self.logger.info(f'训练数据加载器 共{len(self.train_dataloader)}')
        self.valid_dataloader = None
        if 'eval' in mode and self.cfg['Eval']:
            self.valid_dataloader = build_dataloader(self.cfg, 'Eval', self.logger)
            self.logger.info(f'评估训练加载器 共{len(self.valid_dataloader)}')
        # 后处理
        self.post_process_class = build_post_process(self.cfg['PostProcess'])
        # 更新识别模型的输出通道数
        update_rec_head_out_channels(self.cfg, self.post_process_class)
        # 构建模型
        self.model = build_model(self.cfg['Architecture'])
        # 将模型转移到GPU上
        self.model = self.model.to(self.device)
        # 是否使用同步BN
        use_sync_bn = self.cfg["Global"].get("use_sync_bn", False)
        if use_sync_bn:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.logger.info('同步BN')

        # 构建损失函数
        self.loss_class = build_loss(self.cfg['Loss'])

        # 初始化优化器和学习率调度器
        self.optimizer, self.lr_scheduler = None, None
        if self.train_dataloader is not None:
            # 构建优化器
            self.optimizer, self.lr_scheduler = build_optimizer(
                self.cfg['Optimizer'],
                self.cfg['LRScheduler'],
                epochs=self.cfg['Global']['epoch_num'],
                step_each_epoch=len(self.train_dataloader),
                model=self.model)
        # 评估
        self.eval_class = build_metric(self.cfg['Metric'])
        # 加载保存的参数
        self.status = load_ckpt(self.model, self.cfg, self.optimizer, self.lr_scheduler)
        # 多卡分布式
        if self.cfg['Global']['distributed']:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, [self.local_rank], find_unused_parameters=True)

        # 混合精度训练
        self.scaler = torch.cuda.amp.GradScaler() if self.cfg['Global'].get('use_amp', False) else None
        # 日志
        self.logger.info(f'启动 Torch版本：{torch.__version__} cuda版本： {self.device}')


    # 设置随机数种子
    def set_random_seed(self, seed):
        # 为CPU设置随机种子
        torch.manual_seed(seed)  
        # cuda
        if self.device.type == 'cuda':
            # 设置 cudnn 后端为基于当前输入大小的自动调整模式
            torch.backends.cudnn.benchmark = True
            # 为当前GPU设置随机种子
            torch.cuda.manual_seed(seed)  
            # 为所有GPU设置随机种子
            torch.cuda.manual_seed_all(seed)  
        # 设置 Python 内置的随机数生成器的种子
        random.seed(seed)
        # 设置 NumPy 随机数生成器的种子
        np.random.seed(seed)

    # 设置cuda
    def set_device(self, device):
        if device == 'gpu' and torch.cuda.is_available():
            device = torch.device(f"cuda:{self.local_rank}")
        else:
            device = torch.device("cpu")
        self.device = device

    def train(self):
        cal_metric_during_train = self.cfg['Global'].get('cal_metric_during_train', False)
        log_smooth_window = self.cfg['Global']['log_smooth_window']
        epoch_num = self.cfg['Global']['epoch_num']
        print_batch_step = self.cfg['Global']['print_batch_step']
        eval_epoch_step = self.cfg['Global'].get('eval_epoch_step', 1)

        start_eval_epoch = 0
        # 验证集
        if self.valid_dataloader is not None:
            if type(eval_epoch_step) == list and len(eval_epoch_step) >= 2:
                start_eval_epoch = eval_epoch_step[0]
                eval_epoch_step = eval_epoch_step[1]
                if len(self.valid_dataloader) == 0:
                    start_eval_epoch = 1e111
                    self.logger.info('没有评估数据集')
                # self.logger.info(f"在{start_eval_epoch}epoch训练后，每{eval_epoch_step}epoch评估一次")
        else:
            start_eval_epoch = 1e111

        global_step = self.status.get('global_step', 0)
        start_epoch = self.status.get('epoch', 1)
        best_metric = self.status.get('metrics', {})
        # 评估实例
        if self.eval_class.main_indicator not in best_metric:
            best_metric[self.eval_class.main_indicator] = 0
        # 训练统计
        train_stats = TrainingStats(log_smooth_window, ['lr'])
        # 训练模型
        self.model.train()

        total_samples = 0
        train_reader_cost = 0.0
        train_batch_cost = 0.0
        reader_start = time.time()
        eta_meter = AverageMeter()

        for epoch in range(start_epoch, epoch_num + 1):
            # 数据集重置
            if self.train_dataloader.dataset.need_reset:
                self.train_dataloader = build_dataloader(self.cfg, 'Train', self.logger)

            for idx, batch in enumerate(self.train_dataloader):
                batch = [t.to(self.device) for t in batch]
                self.optimizer.zero_grad()
                train_reader_cost += time.time() - reader_start

                if self.scaler:
                    with torch.cuda.amp.autocast():
                        preds = self.model(batch[0], data=batch[1:])
                        loss = self.loss_class(preds, batch)
                    self.scaler.scale(loss['loss']).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    preds = self.model(batch[0], data=batch[1:])
                    loss = self.loss_class(preds, batch)
                    avg_loss = loss['loss']
                    avg_loss.backward()
                    self.optimizer.step()

                # 计算评估指标
                if cal_metric_during_train: 
                    post_result = self.post_process_class(preds, batch)
                    self.eval_class(post_result, batch)
                    metric = self.eval_class.get_metric()
                    train_stats.update(metric)
                # 计算时间
                train_batch_time = time.time() - reader_start
                train_batch_cost += train_batch_time
                # 更新预估的剩余训练时间
                eta_meter.update(train_batch_time)
                global_step += 1
                total_samples += len(batch[0])
                # 更新学习率
                self.lr_scheduler.step()

                # 记录训练日志
                # 损失值
                stats = {
                    k: float(v) if v.shape == [] else v.detach().cpu().numpy().mean()
                    for k, v in loss.items()
                }
                # 记录学习率
                stats['lr'] = self.lr_scheduler.get_last_lr()[0]
                # 记录统计
                train_stats.update(stats)
                # TensorBoard 
                if self.writer is not None:
                    for k, v in train_stats.get().items():
                        self.writer.add_scalar(f'TRAIN/{k}', v, global_step)
                # 打印日志
                if self.local_rank == 0 and (
                        (global_step > 0 and global_step % print_batch_step == 0) or
                        (idx >= len(self.train_dataloader) - 1)):
                    # 获取训练统计信息的日志
                    logs = train_stats.log()
                    # 计算剩余时间
                    eta_sec = ((epoch_num + 1 - epoch) * len(self.train_dataloader) - idx - 1) * eta_meter.avg
                    eta_sec_format = str(datetime.timedelta(seconds=int(eta_sec)))
                    # 日志字符串
                    strs = (f'epoch: [{epoch}/{epoch_num}], global_step: {global_step}, {logs}, '
                            f'avg_reader_cost: {train_reader_cost / print_batch_step:.5f} s, '
                            f'avg_batch_cost: {train_batch_cost / print_batch_step:.5f} s, '
                            f'avg_samples: {total_samples / print_batch_step}, '
                            f'ips: {total_samples / train_batch_cost:.5f} samples/s, '
                            f'eta: {eta_sec_format}')
                    # 打印日志
                    self.logger.info(strs)
                    total_samples = 0
                    train_reader_cost = 0.0
                    train_batch_cost = 0.0

                reader_start = time.time()
            
            # 模型评估
            if self.local_rank == 0 and epoch > start_eval_epoch and (epoch - start_eval_epoch) % eval_epoch_step == 0:
                # 评估
                cur_metric = self.eval()
                cur_metric_str = f"当前评估, {', '.join(['{}: {}'.format(k, v) for k, v in cur_metric.items()])}"
                self.logger.info(cur_metric_str)

                # TensorBoard
                if self.writer is not None:
                    for k, v in cur_metric.items():
                        if isinstance(v, (float, int)):
                            self.writer.add_scalar(f'EVAL/{k}', cur_metric[k], global_step)
                # 更新最新的评估指标
                if cur_metric[self.eval_class.main_indicator] >= best_metric[self.eval_class.main_indicator]:
                    best_metric.update(cur_metric)
                    best_metric['best_epoch'] = epoch
                    # TensorBoard
                    if self.writer is not None:
                        self.writer.add_scalar(f'EVAL/best_{self.eval_class.main_indicator}',
                                               best_metric[self.eval_class.main_indicator], global_step)
                    # 保存最佳模型
                    save_ckpt(self.model, self.cfg, self.optimizer, self.lr_scheduler, epoch, global_step, best_metric, is_best=True)
                best_str = f"最优评估, {', '.join(['{}: {}'.format(k, v) for k, v in best_metric.items()])}"
                self.logger.info(best_str)

            # 保存模型
            if self.local_rank == 0:
                save_ckpt(self.model, self.cfg, self.optimizer, self.lr_scheduler, epoch, global_step, best_metric, is_best=False)

        best_str = f"最优指标, {', '.join(['{}: {}'.format(k, v) for k, v in best_metric.items()])}"
        self.logger.info(best_str)
        # 关闭TensorBoard
        if self.writer is not None:
            self.writer.close()
        # 释放GPU
        if torch.cuda.device_count() > 1:
            torch.distributed.destroy_process_group()

    def eval(self):
        # 评估模型
        self.model.eval()
        with torch.no_grad():
            total_frame = 0.0
            total_time = 0.0
            # 进度条
            pbar = tqdm(
                total=len(self.valid_dataloader),
                desc='eval model:',
                position=0,
                leave=True)
            sum_images = 0
            # 遍历评估数据集
            for idx, batch in enumerate(self.valid_dataloader):
                batch = [t.to(self.device) for t in batch]
                start = time.time()
                # 混合精度
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        preds = self.model(batch[0], data=batch[1:])
                else:
                    preds = self.model(batch[0], data=batch[1:])
                # 计算时间
                total_time += time.time() - start
                # 后处理
                post_result = self.post_process_class(preds, batch)
                # 评估
                self.eval_class(post_result, batch)
                # 更新进度条
                pbar.update(1)
                total_frame += len(batch[0])
                sum_images += 1
            # 获取评估指标
            metric = self.eval_class.get_metric()
        pbar.close()
        # 设置为训练模式
        self.model.train()
        metric['fps'] = total_frame / total_time
        return metric

    # def test_dataloader(self):
    #     starttime = time.time()
    #     count = 0
    #     try:
    #         for data in self.train_dataloader:
    #             count += 1
    #             if count % 1 == 0:
    #                 batch_time = time.time() - starttime
    #                 starttime = time.time()
    #                 self.logger.info(f"reader: {count}, {data[0].shape}, {batch_time}")
    #     except:
    #         import traceback
    #         self.logger.info(traceback.format_exc())
    #     self.logger.info(f"finish reader: {count}, Success!")
