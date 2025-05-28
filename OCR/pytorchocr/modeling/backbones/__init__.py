__all__ = ['build_backbone']

def build_backbone(config, model_type):
    if model_type == 'det':
        from .det_resnet import ResNet_det
        support_dict = ['ResNet_det']
    elif model_type == 'rec':
        from .rec_resnet import ResNet_rec
        support_dict = ['ResNet_rec']

    module_name = config.pop('name')
    assert module_name in support_dict, Exception('model error backbone')
    module_class = eval(module_name)(**config)
    return module_class