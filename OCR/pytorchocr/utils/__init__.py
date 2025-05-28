from .img_aug import *


def transform(data, ops=None):
    if ops is None:
        ops = []
    # 遍历操作列表
    for op in ops:
        data = op(data)
        if data is None:
            return None
    return data

def preprocess(op_param_list):

    assert isinstance(op_param_list, list), ('error preprocess')
    ops = []
    #  遍历操作参数列表
    for operator in op_param_list:
        assert isinstance(operator,dict) and len(operator) == 1, "error preprocess"
        # 操作名称
        operator_name = list(operator)[0]
        # 操作参数
        param = {} if operator[operator_name] is None else operator[operator_name]
        # 根据操作名称创建操作
        op = eval(operator_name)(**param)
        # 添加到操作列表
        ops.append(op)
    return ops