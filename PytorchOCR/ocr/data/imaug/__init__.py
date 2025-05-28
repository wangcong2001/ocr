
from .iaa_augment import IaaAugment # todo
from .make_border_map import MakeBorderMap# todo
from .make_shrink_map import MakeShrinkMap# todo
from .random_crop_data import EastRandomCropData, RandomCropImgMask# todo


# todo RecConAug RecAug RecResizeImg
from .rec_img_aug import BaseDataAugmentation, RecAug, RecConAug, RecResizeImg, \
    SRNRecResizeImg, GrayRecResizeImg, SARRecResizeImg, PRENResizeImg, \
    ABINetRecResizeImg, SVTRRecResizeImg, VLRecResizeImg, SPINRecResizeImg, RobustScannerRecResizeImg, \
    RFLRecResizeImg
from .operators import * # todo
from .label_ops import * # todo



def transform(data, ops=None):
    if ops is None:
        ops = []
    for op in ops:
        data = op(data)
        if data is None:
            return None
    return data


def create_operators(op_param_list, global_config=None):
    assert isinstance(op_param_list, list), ('需要一个操作符列表')
    ops = []
    for operator in op_param_list:
        assert isinstance(operator, dict) and len(operator) == 1, "配置文件错误"
        # 获取操作符名称
        op_name = list(operator)[0]
        # 解析参数
        param = {} if operator[op_name] is None else operator[op_name]
        # 更新全局配置
        if global_config is not None:
            param.update(global_config)
        # 实例化
        op = eval(op_name)(**param)
        # 添加进列表
        ops.append(op)
    return ops
