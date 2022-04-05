#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2022/02/16 21:54
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : math_utils.py
# @Software  : PyCharm
import numpy as np
from numpy import dot
from numpy.linalg import norm
import torch
from torch import nn


class EntropyLoss(nn.Module):
    def __init__(self, eps=1e-12):
        super(EntropyLoss, self).__init__()
        self.eps = eps

    def forward(self, x):
        b = x * torch.log(x + self.eps)
        b = -1.0 * b.sum(dim=1)
        b = b.mean()
        return b


class EntropyLossEncap(nn.Module):
    def __init__(self, eps=1e-12):
        super(EntropyLossEncap, self).__init__()
        self.eps = eps
        self.entropy_loss = EntropyLoss(eps)

    def forward(self, input):
        score = feature_map_permute(input)
        ent_loss_val = self.entropy_loss(score)
        return ent_loss_val


def feature_map_permute(input):
    s = input.data.shape
    l = len(s)

    # permute feature channel to the last:
    # NxCxDxHxW --> NxDxHxW x C
    if l == 2:
        x = input  # NxC
    elif l == 3:
        x = input.permute(0, 2, 1)
    elif l == 4:
        x = input.permute(0, 2, 3, 1)
    elif l == 5:
        x = input.permute(0, 2, 3, 4, 1)
    else:
        x = []
        print('wrong feature map size')
    x = x.contiguous()
    # NxDxHxW x C --> (NxDxHxW) x C
    x = x.view(-1, s[1])
    return x


def cos_sim(A, B):
    return dot(A, B) / (norm(A) * norm(B))