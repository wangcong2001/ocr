__all__ = ['build_neck']


def build_neck(config):
    from .db_fpn import RSEFPN
    from .rnn import SequenceEncoder
    support_dict = [
        'SequenceEncoder', 'RSEFPN'
    ]

    module_name = config.pop('name')
    assert module_name in support_dict, Exception("Neck层错误")

    module_class = eval(module_name)(**config)
    return module_class
