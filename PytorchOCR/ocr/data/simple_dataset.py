import numpy as np
import cv2
import math
import os
import json
import random
import traceback
from torch.utils.data import Dataset
from .imaug import transform, create_operators


class SimpleDataSet(Dataset):
    def __init__(self, config, mode, logger, seed=None):
        super(SimpleDataSet, self).__init__()
        self.logger = logger
        self.mode = mode.lower()
        # 获取当前参数配置
        global_config = config['Global']
        dataset_config = config[mode]['dataset']
        loader_config = config[mode]['loader']
        # 获取分隔符
        self.delimiter = dataset_config.get('delimiter', '\t')
        # 获取标签列表
        label_file_list = dataset_config.pop('label_file_list')
        # 获取数据源数量
        data_source_num = len(label_file_list)
        # 获取样本比例列表
        ratio_list = dataset_config.get("ratio_list", 1.0)
        # 转换为列表
        if isinstance(ratio_list, (float, int)):
            ratio_list = [float(ratio_list)] * int(data_source_num)

        assert len(ratio_list) == data_source_num, "数据源数量错误"
        # 获取数据目录
        self.data_dir = dataset_config['data_dir']
        # 是否打乱数据
        self.do_shuffle = loader_config['shuffle']
        # 获取随机种子
        self.seed = seed
        # 记录日志
        logger.info(f"初始化数据集: {label_file_list}")
        # 从指定的文件列表中读取数据，并根据给定的比例列表对数据进行采样
        self.data_lines = self.get_image_info_list(label_file_list, ratio_list)
        # 创建索引列表
        self.data_idx_order_list = list(range(len(self.data_lines)))
        # 是否打乱数据
        if self.mode == "train" and self.do_shuffle:
            self.shuffle_data_random()
        # 设置随机数
        self.set_epoch_as_seed(self.seed, dataset_config)
        # 创建操作符列表
        self.ops = create_operators(dataset_config['transforms'], global_config)
        # 获取扩展操作索引
        self.ext_op_transform_idx = dataset_config.get("ext_op_transform_idx",)
        # 是否需要重置数据集
        self.need_reset = True in [x < 1 for x in ratio_list]

    def set_epoch_as_seed(self, seed, dataset_config):
        # 在训练模式下
        if self.mode == 'train':
            try:
                # 获取MakeBorderMap
                border_map_id = [index
                    for index, dictionary in enumerate(dataset_config['transforms'])
                    if 'MakeBorderMap' in dictionary][0]
                # 获取MakeShrinkMap
                shrink_map_id = [index
                    for index, dictionary in enumerate(dataset_config['transforms'])
                    if 'MakeShrinkMap' in dictionary][0]
                # 设置随机种子
                dataset_config['transforms'][border_map_id]['MakeBorderMap'][
                    'epoch'] = seed if seed is not None else 0
                dataset_config['transforms'][shrink_map_id]['MakeShrinkMap'][
                    'epoch'] = seed if seed is not None else 0
            except Exception as E:
                return

    def get_image_info_list(self, file_list, ratio_list):
        # 转换成列表
        if isinstance(file_list, str):
            file_list = [file_list]
        # 数据列表
        data_lines = []
        # 遍历文件列表
        for idx, file in enumerate(file_list):
            # 打开文件
            with open(file, "rb") as f:
                lines = f.readlines()
                # 随机采样
                if self.mode == "train" or ratio_list[idx] < 1.0:
                    random.seed(self.seed)
                    lines = random.sample(lines,
                                          round(len(lines) * ratio_list[idx]))
                data_lines.extend(lines)
        return data_lines

    def shuffle_data_random(self):
        # 设置随机数种子
        random.seed(self.seed)
        # 打乱列表数据
        random.shuffle(self.data_lines)
        return
    # 解析文件名列表
    def _try_parse_filename_list(self, file_name):
        # multiple images -> one gt label
        # 文件名是否为列表形式
        if len(file_name) > 0 and file_name[0] == "[":
            try:
                # 解析json
                info = json.loads(file_name)
                # 获取为文件名
                file_name = random.choice(info)
            except:
                pass
        return file_name


    def get_ext_data(self):
        ext_data_num = 0
        # 操作序列
        for op in self.ops:
            # 是否存在扩展数据
            if hasattr(op, 'ext_data_num'):
                # 获取扩展数据数量
                ext_data_num = getattr(op, 'ext_data_num')
                break
        # 获取加载数据操作
        load_data_ops = self.ops[:self.ext_op_transform_idx]
        # 扩展数据列表
        ext_data = []

        while len(ext_data) < ext_data_num:
            # 获取随机索引
            file_idx = self.data_idx_order_list[np.random.randint(self.__len__())]
            # 获取数据行
            data_line = self.data_lines[file_idx]
            # 解码数据行
            data_line = data_line.decode('utf-8')
            # 分割数据行
            substr = data_line.strip("\n").split(self.delimiter)
            # 获取文件名
            file_name = substr[0]
            # 解析文件名列表
            file_name = self._try_parse_filename_list(file_name)
            # 获取label
            label = substr[1]
            # 获取图片路径
            img_path = os.path.join(self.data_dir, file_name)
            # 创建数据字典
            data = {'img_path': img_path, 'label': label}
            # 判断图片是否存在
            if not os.path.exists(img_path):
                continue
            # 读取图片
            with open(data['img_path'], 'rb') as f:
                img = f.read()
                data['image'] = img
            # 对图片进行操作变换
            data = transform(data, load_data_ops)
            # 判断是否将数据添加进列表
            if data is None:
                continue
            if 'polys' in data.keys():
                if data['polys'].shape[1] != 4:
                    continue
            ext_data.append(data)
        return ext_data

    def __getitem__(self, idx):
        # 获取索引文件名
        file_idx = self.data_idx_order_list[idx]
        # 获取数据行 img lable
        data_line = self.data_lines[file_idx]
        try:
            # 解码数据行
            data_line = data_line.decode('utf-8')
            # 分割数据行
            substr = data_line.strip("\n").split(self.delimiter)
            # 获取文件名
            file_name = substr[0]
            # 解析文件名列表
            file_name = self._try_parse_filename_list(file_name)
            # 获取label
            label = substr[1]
            # 获取图片路径
            img_path = os.path.join(self.data_dir, file_name)
            # 创建数据字典
            data = {'img_path': img_path, 'label': label}
            # 判断图片是否存在
            if not os.path.exists(img_path):
                raise Exception("{} 图片不存在".format(img_path))
            # 读取图片
            with open(data['img_path'], 'rb') as f:
                img = f.read()
                data['image'] = img
            # 获取扩展数据
            data['ext_data'] = self.get_ext_data()
            # 对数据进行变换操作
            outs = transform(data, self.ops)
        except:
            self.logger.error(
                "数据 {}, 错误: {}".format(
                    data_line, traceback.format_exc()))
            outs = None
        if outs is None:
            # 训练模式，则随机选择另一个索引，调用 __getitem__ 方法，直到返回有效的数据。
            # 评估模式，则根据当前索引计算下一个索引，并返回对应的数据。
            rnd_idx = np.random.randint(self.__len__()) if self.mode == "train" else (idx + 1) % self.__len__()
            return self.__getitem__(rnd_idx)
        return outs
    # 获取数据集长度
    def __len__(self):
        return len(self.data_idx_order_list)