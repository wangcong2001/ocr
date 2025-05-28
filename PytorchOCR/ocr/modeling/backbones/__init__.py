__all__ = ["build_backbone"]

# 构建主干网络
def build_backbone(config, model_type):
    if model_type == "det":
        from .det_mobilenet_v3 import MobileNetV3
        support_dict = [ 'MobileNetV3']
    elif model_type == "rec":
        from .rec_mobilenet_v3 import MobileNetV3
        support_dict = ['MobileNetV3']
    else:
        raise NotImplementedError

    module_name = config.pop('name')
    assert module_name in support_dict, Exception("主干网络错误")
    # 实例化主干网络
    module_class = eval(module_name)(**config)
    return module_class
