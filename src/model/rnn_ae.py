#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2021/12/08 15:12
# @Author    : Junhyung Kwon
# @Site      :
# @File      : RNN_AE.py
# @Software  : PyCharm
import torch
import torch.nn.functional as F
from torch import nn


class HSCRNNEncoder(nn.Module):
    def __init__(self, in_dim=32,
                 hidden_dim=16,
                 rep_dim =256,
                 num_layers=2,
                 window_size=128):
        super(HSCRNNEncoder, self).__init__()

        self.rnn_layer = nn.GRU(in_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc_layer = nn.Sequential(
            nn.Linear(hidden_dim * window_size, 512),
            nn.LeakyReLU(),
            nn.Linear(512, rep_dim),
            nn.LeakyReLU()
        )

    def forward(self, X):
        out, h = self.rnn_layer(X)
        out = out.reshape((out.size(0), -1))
        final_output = self.fc_layer(out)

        return final_output


class RNNEncoder(nn.Module):
    def __init__(self, rnn_type, input_dim, encoder_dim, hidden_dim, num_layers, dropout=0, batch_first=True):
        super(RNNEncoder, self).__init__()
        assert rnn_type in ['GRU', 'LSTM', 'RNN'], "RNN type must be one of 'GRU', 'LSTM', 'RNN'"
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.rnn_type = rnn_type
        self.rnn_layer = getattr(nn, rnn_type)(input_dim, hidden_dim, num_layers, batch_first=batch_first)
        self.fc = nn.Linear(hidden_dim, encoder_dim)

        self.dropout = nn.Dropout(p=dropout)

    def init_hidden(self, batch_size, device=None):
        if device is None:
            device = 'cpu'
        if self.rnn_type == 'LSTM':
            h = nn.init.kaiming_normal_(
                torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device, dtype=torch.float32))
            c = nn.init.kaiming_normal_(
                torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device, dtype=torch.float32))

            return (h.to(device), c.to(device))
        else:
            return nn.init.kaiming_normal_(
                torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device, dtype=torch.float32))

    def forward(self, x, h):
        rnn_out, h = self.rnn_layer(x, h)
        out = rnn_out.contiguous().view(-1, self.hidden_dim)
        out = self.dropout(out)
        out = F.relu(self.fc(out))
        return out


class RNNEncoderDSVDD(nn.Module):
    def __init__(self, rnn_type, input_dim, encoder_dim, hidden_dim, num_layers, device, dropout=0, batch_first=True):
        super(RNNEncoderDSVDD, self).__init__()
        assert rnn_type in ['GRU', 'LSTM', 'RNN'], "RNN type must be one of 'GRU', 'LSTM', 'RNN'"
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.rnn_type = rnn_type
        self.rnn_layer = getattr(nn, rnn_type)(input_dim, hidden_dim, num_layers, batch_first=batch_first)
        self.fc = nn.Linear(hidden_dim, encoder_dim)
        self.device = device

        self.dropout = nn.Dropout(p=dropout)

    def init_hidden(self, batch_size, device=None):
        if device is None:
            device = 'cpu'
        if self.rnn_type == 'LSTM':
            h = nn.init.kaiming_normal_(
                torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device, dtype=torch.float32))
            c = nn.init.kaiming_normal_(
                torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device, dtype=torch.float32))

            return (h.to(device), c.to(device))
        else:
            return nn.init.kaiming_normal_(
                torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device, dtype=torch.float32))

    def forward(self, x):
        x = x.float()
        batch_size = x.size(0)
        h = self.init_hidden(batch_size=batch_size, device=self.device)

        rnn_out, h = self.rnn_layer(x, h)
        out = rnn_out.contiguous().view(-1, self.hidden_dim)
        out = self.dropout(out)
        out = F.relu(self.fc(out))
        return out


class RNNDecoder(nn.Module):
    def __init__(self, rnn_type, input_dim, encoder_dim, hidden_dim, num_layers, dropout=0, batch_first=True):
        super(RNNDecoder, self).__init__()
        assert rnn_type in ['GRU', 'LSTM', 'RNN'], "RNN type must be one of 'GRU', 'LSTM', 'RNN'"
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.rnn_type = rnn_type
        self.rnn_layer = getattr(nn, rnn_type)(hidden_dim, input_dim, num_layers, batch_first=batch_first)
        self.fc = nn.Linear(encoder_dim, hidden_dim)

        self.dropout = nn.Dropout(p=dropout)

    def init_hidden(self, batch_size, device=None):
        if device is None:
            device = 'cpu'
        if self.rnn_type == 'LSTM':
            h = nn.init.kaiming_normal_(
                torch.zeros(self.num_layers, batch_size, self.input_dim, device=device, dtype=torch.float32))
            c = nn.init.kaiming_normal_(
                torch.zeros(self.num_layers, batch_size, self.input_dim, device=device, dtype=torch.float32))

            return (h.to(device), c.to(device))
        else:
            return nn.init.kaiming_normal_(
                torch.zeros(self.num_layers, batch_size, self.input_dim, device=device, dtype=torch.float32))

    def forward(self, x, h, batch_size):
        out = F.relu(self.fc(x))
        out = self.dropout(out)
        out = out.view(batch_size, -1, self.hidden_dim)
        out, h = self.rnn_layer(out, h)

        return out


class RNNAE(nn.Module):
    def __init__(self,
                 rnn_type='GRU',
                 input_dim=661,
                 encoder_dim=64,
                 hidden_dim=256,
                 num_layers=1,
                 dropout=0,
                 batch_first=True,
                 device='cpu'):
        super(RNNAE, self).__init__()
        self.device = device
        self.encoder_dim = encoder_dim
        self.encoder = RNNEncoder(rnn_type, input_dim, encoder_dim, hidden_dim, num_layers, dropout, batch_first)
        self.decoder = RNNDecoder(rnn_type, input_dim, encoder_dim, hidden_dim, num_layers, dropout, batch_first)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.GRU) or isinstance(m, nn.LSTM) or isinstance(m, nn.RNN):
                for param in m.parameters():
                    if len(param.shape) >= 2:
                        nn.init.kaiming_normal_(param.data)
                    else:
                        nn.init.normal_(param.data)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def encode(self, x, transpose=False):
        if transpose:
            x = x.transpose(1, 2)
        batch_size = x.size(0)
        h1 = self.encoder.init_hidden(batch_size=batch_size, device=self.device)
        z = self.encoder(x, h1)
        z = z.view(batch_size, -1, self.encoder_dim)
        return z

    def forward(self, x, transpose=False):
        if transpose:
            x = x.transpose(1, 2)
        batch_size = x.size(0)
        h1 = self.encoder.init_hidden(batch_size=batch_size, device=self.device)
        h2 = self.decoder.init_hidden(batch_size=batch_size, device=self.device)
        z = self.encoder(x, h1)
        x_ = self.decoder(z, h2, batch_size)
        if transpose:
            x_ = x_.transpose(1, 2)
        return x_
