import sys
import six
import cv2
import numpy as np

# 图像预处理 归一化
class NormalizeImage(object):

    def __init__(self, scale=None, mean=None, std=None, order='chw', **kwargs):
        if isinstance(scale, str):  # 将缩放因子转换为浮点数
            scale = eval(scale)
        # scale = 1.0 / 255.0 默认值
        self.scale = np.float32(scale if scale is not None else 1.0 / 255.0)
        # 均值 = [0.485, 0.456, 0.406] 默认值
        mean = mean if mean is not None else [0.485, 0.456, 0.406]
        # 标准差 = [0.229, 0.224, 0.225] 默认值
        std = std if std is not None else [0.229, 0.224, 0.225]
        # 匹配通道
        shape = (3, 1, 1) if order == 'chw' else (1, 1, 3)
        # 将均值和标准差转换为数组
        self.mean = np.array(mean).reshape(shape).astype('float32')
        self.std = np.array(std).reshape(shape).astype('float32')

    def __call__(self, data):
        # 获取图像
        img = data['image']
        from PIL import Image
        # 如果是PIL图像转为数组
        if isinstance(img, Image.Image):
            img = np.array(img)
        assert isinstance(img, np.ndarray), "error normalization"
        # 归一化
        data['image'] = (
            img.astype('float32') * self.scale - self.mean) / self.std
        return data
    
# 图像预处理 转换维度 hwc->chw
class ToCHWImage(object):

    def __call__(self, data):
        img = data['image']
        from PIL import Image
        if isinstance(img, Image.Image):
            img = np.array(img)
        # 转换维度
        data['image'] = img.transpose((2, 0, 1))
        return data

# 图像预处理 保留字段 'keep_keys': ['image', 'shape']
class KeepKeys(object):
    def __init__(self, keep_keys, **kwargs):
        self.keep_keys = keep_keys
    def __call__(self, data):
        data_list = []
        for key in self.keep_keys:
            data_list.append(data[key])
        return data_list

# 图像预处理 调整图像大小
class DetResizeForTest(object):
    def __init__(self, **kwargs):
        super(DetResizeForTest, self).__init__()
        if 'limit_side_len' in kwargs:
            self.limit_side_len = kwargs['limit_side_len']
            self.limit_type = kwargs.get('limit_type', 'min')
        else:
            self.limit_side_len = 736
            self.limit_type = 'min'

    def __call__(self, data):
        # 获取原始数据
        img = data['image']
        src_h, src_w, _ = img.shape
        # 调整图像大小
        img, [ratio_h, ratio_w] = self.resize_image(img)
        # 保存数据
        data['image'] = img
        data['shape'] = np.array([src_h, src_w, ratio_h, ratio_w])
        return data
    
    def resize_image(self, img):
        # 获取图像的高度和宽度
        limit_side_len = self.limit_side_len
        h, w, c = img.shape

        # 计算大小比例
        if self.limit_type == 'max':
            # 如果最大边大于限制边长
            if max(h, w) > limit_side_len:
                # 如果高度大于宽度
                if h > w:
                    ratio = float(limit_side_len) / h
                else:
                    ratio = float(limit_side_len) / w
            else:
                ratio = 1.
        elif self.limit_type == 'min':
            # 如果最小边小于限制边长
            if min(h, w) < limit_side_len:
                if h < w:
                    ratio = float(limit_side_len) / h
                else:
                    ratio = float(limit_side_len) / w
            else:
                ratio = 1.
        # 计算调整后的高度和宽度
        resize_h = int(h * ratio)
        resize_w = int(w * ratio)
        # 调整大小确保是32的倍数
        resize_h = max(int(round(resize_h / 32) * 32), 32)
        resize_w = max(int(round(resize_w / 32) * 32), 32)
        # 调整图像大小
        try:
            if int(resize_w) <= 0 or int(resize_h) <= 0:
                return None, (None, None)
            img = cv2.resize(img, (int(resize_w), int(resize_h)))
        except:
            print(img.shape, resize_w, resize_h)
            sys.exit(0)
        # 计算比例
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)
        return img, [ratio_h, ratio_w]