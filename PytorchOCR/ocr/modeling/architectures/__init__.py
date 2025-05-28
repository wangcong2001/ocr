import copy
import importlib

from .base_model import BaseModel

__all__ = ["build_model"]

def build_model(config):
    config = copy.deepcopy(config)
    arch = BaseModel(config)
    return arch