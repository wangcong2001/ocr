__all__ = ['build_neck']

def build_neck(config):
    from .db_fpn import DBFPN
    from .rnn import SequenceEncoder
    support_dict = ['DBFPN','SequenceEncoder']
    module_name = config.pop('name')
    assert module_name in support_dict, Exception('model error neck')
    module_class = eval(module_name)(**config)
    return module_class