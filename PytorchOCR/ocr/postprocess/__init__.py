
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy

__all__ = ['build_post_process']

from .db_postprocess import DBPostProcess
from .rec_postprocess import CTCLabelDecode

# todo DBPostProcess CTCLabelDecode
def build_post_process(config, global_config=None):
    support_dict = ['DBPostProcess', 'CTCLabelDecode']
    # 复制config
    config = copy.deepcopy(config)
    # 获取后处理模块名称
    module_name = config.pop('name')
    # 后处理为空
    if module_name == "None":
        return
    # 如果有全局配置，更新配置
    if global_config is not None:
        config.update(global_config)
    
    assert module_name in support_dict, Exception("后处理错误")
    # 实例化
    module_class = eval(module_name)(**config)
    return module_class
