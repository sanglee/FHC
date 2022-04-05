#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2022/02/19 1:55 PM
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : cae.py
# @Software  : PyCharm

from torch import nn
import torch


class CAE(nn.Module):
    def __init__(self, in_dim, num_layers=3, hidden_dim=512, filter_size=7, stride=1):
        super(CAE, self).__init__()
        assert hidden_dim is not None, 'hidden dimension must be subjected'

        encoder = []
        prev_dim = in_dim
        for i in range(num_layers):
            encoder.append(nn.Conv1d(prev_dim, hidden_dim, filter_size, stride, filter_size // 2))
            encoder.append(nn.LeakyReLU())
            prev_dim = hidden_dim
            hidden_dim //= 2

        self.encoder = nn.Sequential(*encoder)

        decoder = []
        hidden_dim = prev_dim * 2
        for i in range(num_layers - 1):
            decoder.append(nn.ConvTranspose1d(prev_dim, hidden_dim, filter_size, stride, filter_size // 2))
            decoder.append(nn.LeakyReLU())
            prev_dim = hidden_dim
            hidden_dim *= 2

        decoder.append(nn.ConvTranspose1d(prev_dim, in_dim, filter_size, stride, filter_size // 2))
        decoder.append(nn.LeakyReLU())

        self.decoder = nn.Sequential(*decoder)

    def encode(self, input, transpose=True):
        if transpose:
            input = input.transpose(1, 2)

        out = self.encoder(input)
        return out.transpose(1, 2)

    def forward(self, input, transpose=True):
        if transpose:
            input = input.transpose(1, 2)

        out = self.encoder(input)
        out = self.decoder(out)
        out = out.transpose(1, 2)

        return out
