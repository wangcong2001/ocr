import torch
import torch.nn as nn
import torch.nn.functional as F




class Activation(nn.Module):
    def __init__(self, act_type, inplace=True):
        super(Activation, self).__init__()
        act_type = act_type.lower()
        if act_type == 'relu':
            self.act = nn.ReLU(inplace=inplace)
    def forward(self, inputs):
        return self.act(inputs)