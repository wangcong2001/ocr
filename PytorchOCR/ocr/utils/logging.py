import os
import sys
import logging
import functools
import torch
import torch.distributed as dist

logger_initialized = {}

# 创建缓存
@functools.lru_cache()
def get_logger(name='ocr', log_file=None, log_level=logging.DEBUG):
    # 获取logger
    logger = logging.getLogger(name)
    # 如果logger已经初始化过了，直接返回
    if name in logger_initialized:
        return logger
    for logger_name in logger_initialized:
        # name的前缀是否是logger_name
        if name.startswith(logger_name):
            return logger
    # 格式化
    formatter = logging.Formatter(
        '[%(asctime)s] %(name)s %(levelname)s: %(message)s',
        datefmt="%Y/%m/%d %H:%M:%S")
    # 创建流处理器重定向到标准输出
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # 获取本地的GPUID
    rank = int(os.environ['LOCAL_RANK']) if 'LOCAL_RANK' in os.environ else 0
    # 在主进程中
    if log_file is not None and rank==0:
        # 获取文件夹
        log_file_folder = os.path.split(log_file)[0]
        os.makedirs(log_file_folder, exist_ok=True)
        # 写入日志
        file_handler = logging.FileHandler(log_file, 'a')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    # 主进程日志级别为log_level=logging.DEBUG
    if rank==0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)
    # 初始化完成
    logger_initialized[name] = True
    # 不再传递消息
    logger.propagate = False
    return logger
