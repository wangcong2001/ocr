import re

import numpy as np
import torch
from torch.nn import functional as F

# 解码器
class BaseRecLabelDecode(object):

    def __init__(self, character_dict_path=None, use_space_char=False):
        # 起始
        self.beg_str = "sos"
        # 终止
        self.end_str = "eos"
        self.reverse = False
        # 字符字典
        self.character_str = []
        # 字符字典路径
        if character_dict_path is None:
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_character = list(self.character_str)
        else:
            with open(character_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode('utf-8').strip("\n").strip("\r\n")
                    self.character_str.append(line)
            if use_space_char:
                self.character_str.append(" ")
            # 转换为列表
            dict_character = list(self.character_str)
        # 添加特殊字符
        dict_character = self.add_special_char(dict_character)
        # 索引字典
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

    # def pred_reverse(self, pred):
    #     pred_re = []
    #     c_current = ''
    #     for c in pred:
    #         if not bool(re.search('[a-zA-Z0-9 :*./%+-]', c)):
    #             if c_current != '':
    #                 pred_re.append(c_current)
    #             pred_re.append(c)
    #             c_current = ''
    #         else:
    #             c_current += c
    #     if c_current != '':
    #         pred_re.append(c_current)

    #     return ''.join(pred_re[::-1])

    def add_special_char(self, dict_character):
        return dict_character
    
    # 解码
    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        result_list = []
        # 忽略标记
        ignored_tokens = self.get_ignored_tokens()
        # 批处理大小
        batch_size = len(text_index)
        # 遍历
        for batch_idx in range(batch_size):
            # 初始换数组
            selection = np.ones(len(text_index[batch_idx]), dtype=bool)
            # 是否删除重复字符
            if is_remove_duplicate:
                selection[1:] = text_index[batch_idx][1:] != text_index[batch_idx][:-1]
            # 忽略标记
            for ignored_token in ignored_tokens:
                selection &= text_index[batch_idx] != ignored_token
            # 获取字符列表
            char_list = [
                self.character[text_id]
                for text_id in text_index[batch_idx][selection]
            ]
            # 获取置信度
            if text_prob is not None:
                conf_list = text_prob[batch_idx][selection]
            else:
                conf_list = [1] * len(selection)
            # 没有字符
            if len(conf_list) == 0:
                conf_list = [0]
            # 链接文字
            text = ''.join(char_list)
            # 添加到结果列表
            result_list.append((text, np.mean(conf_list).tolist()))
        return result_list

    def get_ignored_tokens(self):
        return [0] 

# CTC 解码器
class CTCLabelDecode(BaseRecLabelDecode):

    def __init__(self, character_dict_path=None, use_space_char=False, **kwargs):
        super(CTCLabelDecode, self).__init__(character_dict_path, use_space_char)

    def __call__(self, preds, batch=None, *args, **kwargs):
        # 获取预测结果
        preds = preds['res']
        # tensor转换
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        # 提取类别索引 实现解码
        preds_idx = preds.argmax(axis=2)
        # 提取最大置信度
        preds_prob = preds.max(axis=2)
        # 解码
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=True)
        if batch is None:
            return text
        # 将批处理数据的标签解码
        label = self.decode(batch[1].cpu().numpy())
        return text, label

    def add_special_char(self, dict_character):
        dict_character = ['blank'] + dict_character
        return dict_character
