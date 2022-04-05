#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2022/02/13 4:21 PM
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : base_ae.py
# @Software  : PyCharm
from base import BaseNet
from torch import nn


class AutoEncoder(BaseNet):
    def __init__(self, in_dim, hidden_dims):
        super(AutoEncoder, self).__init__()

        self.encoder = Encoder(in_dim, hidden_dims)
        self.decoder = Decoder(in_dim, hidden_dims)
        # encoder = []
        # prev_hd = in_dim
        # for i, hd in enumerate(hidden_dims):
        #     encoder.append(nn.Linear(prev_hd, hd))
        #     prev_hd = hd
        #     encoder.append(nn.BatchNorm1d(hd))
        #     encoder.append(nn.ReLU())
        # self.encoder = nn.Sequential(*encoder)
        #
        # hidden_dims.reverse()
        # hidden_dims.append(in_dim)
        # prev_hd = hidden_dims[0]
        # _ = hidden_dims.pop(0)
        #
        # decoder = []
        # for i, hd in enumerate(hidden_dims):
        #     decoder.append(nn.Linear(prev_hd, hd))
        #     prev_hd = hd
        #     decoder.append(nn.BatchNorm1d(hd))
        #     decoder.append(nn.ReLU())
        # self.decoder = nn.Sequential(*decoder)

    def forward(self, input):
        z = self.encoder(input)
        x_ = self.decoder(z)
        return z, x_


class Encoder(BaseNet):
    def __init__(self, in_dim, hidden_dims):
        super(Encoder, self).__init__()

        encoder = []
        prev_hd = in_dim
        for i, hd in enumerate(hidden_dims):
            encoder.append(nn.Linear(prev_hd, hd))
            prev_hd = hd
            encoder.append(nn.BatchNorm1d(hd))
            encoder.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder)

    def forward(self, input):
        z = self.encoder(input)
        return z


class Decoder(BaseNet):
    def __init__(self, in_dim, hidden_dims):
        super(Decoder, self).__init__()
        hidden_dims.reverse()
        decoder = []
        hidden_dims.append(in_dim)
        prev_hd = hidden_dims[0]
        _ = hidden_dims.pop(0)
        for i, hd in enumerate(hidden_dims):
            decoder.append(nn.Linear(prev_hd, hd))
            prev_hd = hd
            decoder.append(nn.BatchNorm1d(hd))
            decoder.append(nn.ReLU())
        self.decoder = nn.Sequential(*decoder)

    def forward(self, z):
        x = self.decoder(z)
        return x
