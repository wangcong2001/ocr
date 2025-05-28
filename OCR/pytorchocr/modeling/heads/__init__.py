__all__ = ['build_head']

def build_head(config, **kwargs):
    from .det_db_head import DBHead
    from .rec_ctc_head import CTCHead
    support_dict = ['DBHead', 'CTCHead']


    module_name = config.pop('name')
    assert module_name in support_dict, Exception('model error head')
    print(config)
    module_class = eval(module_name)(**config, **kwargs)
    return module_class