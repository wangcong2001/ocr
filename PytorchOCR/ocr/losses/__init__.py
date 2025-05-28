import copy
# det loss
from .det_db_loss import DBLoss
# rec loss
from .rec_multi_loss import MultiLoss

def build_loss(config):
    support_dict = [
        'DBLoss', 'MultiLoss',
    ]
    config = copy.deepcopy(config)
    module_name = config.pop('name')
    module_class = eval(module_name)(**config)
    return module_class
