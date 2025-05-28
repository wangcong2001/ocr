import copy

__all__ = ["build_metric"]

from .det_metric import DetMetric
from .rec_metric import RecMetric


def build_metric(config):
    support_dict = [ "DetMetric", "RecMetric"]

    config = copy.deepcopy(config)
    module_name = config.pop("name")
    assert module_name in support_dict, Exception("评估指标 错误")
    module_class = eval(module_name)(**config)
    return module_class
