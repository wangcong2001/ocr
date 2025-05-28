import copy
__all__ = ['build_post_process']

def build_post_process(config, global_config=None):
    from .db_postprocess import DBPostProcess
    from .rec_postprocess import CTCLabelDecode
    support_dict = [
        'DBPostProcess', 
        'CTCLabelDecode',
    ]
    config = copy.deepcopy(config)
    module_name = config.pop('name')
    if global_config is not None:
        config.update(global_config)
    assert module_name in support_dict, Exception("process error")
    module_class = eval(module_name)(**config)
    return module_class