__all__ = ['build_head']


def build_head(config):
    # det head
    from .det_db_head import DBHead
    # rec head
    from .rec_ctc_head import CTCHead
    from .rec_multi_head import MultiHead
    from .rec_sar_head import SARHead

    support_dict = ['MultiHead', 'SARHead', 'DBHead', 'CTCHead']

    module_name = config.pop('name')
    assert module_name in support_dict, Exception('Head层错误')
    module_class = eval(module_name)(**config)
    # print(config)
    return module_class
