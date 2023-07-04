#!/usr/bin/env python
"""
-------------------------------------------------
   File Name：   dct
   Author :      wenbo
   date：         12/4/2019
   Description :
-------------------------------------------------
   Change Activity:
                   12/4/2019:
-------------------------------------------------
"""
__author__ = 'wenbo'

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from ._dct import dct1, idct1, dct, idct, apply_linear_2d

class LinearDCTModule(nn.Module):
    """Implement any DCT as a linear layer; in practice this executes around
    50x faster on GPU. Unfortunately, the DCT matrix is stored, which will
    increase memory usage.
    :param in_features: size of expected input
    :param type: which dct function in this file to use"""

    def __init__(self, in_features, type, norm=None):
        super(LinearDCTModule, self).__init__()
        self.type = type
        self.N = in_features
        self.norm = norm
        I = torch.eye(self.N)
        if self.type == 'dct1':
            self.weight = dct1(I).data.t()
        elif self.type == 'idct1':
            self.weight = idct1(I).data.t()
        elif self.type == 'dct':
            self.weight = dct(I, norm=self.norm).data.t()
        elif self.type == 'idct':
            self.weight = idct(I, norm=self.norm).data.t()
        # self.register_buffer('weight', kernel)
        # self.weight = kernel

    def forward(self, x: Tensor):
        return F.linear(x, weight=self.weight.to(x.device))

class DCT_Lowfrequency(nn.Module):
    def __init__(self, size=256, fLimit=50):
        super(DCT_Lowfrequency, self).__init__()
        self.fLimit = fLimit
        self.dct = LinearDCTModule(size, type='dct', norm='ortho')
        self.dctTransformer = lambda x: apply_linear_2d(x, self.dct)

    def forward(self, x):
        x = self.dctTransformer(x)
        x = x[:, :, :self.fLimit, :self.fLimit]
        return x
