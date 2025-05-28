import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import OrderedDict
import numpy as np
import cv2
import torch

from OCR.pytorchocr.modeling.architectures.base_model import BaseModel

class Base:
    def __init__(self, config, **kwargs):
        self.config = config
        self.build_net(**kwargs)
        self.net.eval()


    def build_net(self, **kwargs):
        self.net = BaseModel(self.config, **kwargs)

    # 读取权重
    def read_pytorch_weights(self, weights_path):
        if not os.path.exists(weights_path):
            raise FileNotFoundError('{} is not exist.'.format(weights_path))
        weights = torch.load(weights_path)
        return weights
    # 获取输出通道数
    def get_out_channels(self, weights):
        # 判断后缀是否为weight 并且权重矩阵的维度是否为2（判定是否为全连接层）
        if list(weights.keys())[-1].endswith('.weight') and len(list(weights.values())[-1].shape) == 2:
            # 获取权重矩阵的第二维的长度
            out_channels = list(weights.values())[-1].numpy().shape[1]
        else:
            # 获取权重矩阵的第一维的长度
            out_channels = list(weights.values())[-1].numpy().shape[0]
        return out_channels

    # 加载权重字典
    def load_state_dict(self, weights):
        self.net.load_state_dict(weights)
        print('weights is load.')


    # 保存权重
    def save_pytorch_weights(self, weights_path):
        try:
            torch.save(self.net.state_dict(), weights_path, _use_new_zipfile_serialization=False)
        except:
            torch.save(self.net.state_dict(), weights_path) 
        print('model is save: {}'.format(weights_path))

    # 打印权重
    def print_pytorch_state_dict(self):
        print('pytorch:')
        for k,v in self.net.state_dict().items():
            print('{}----{}'.format(k,type(v)))

