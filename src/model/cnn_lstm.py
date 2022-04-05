from torchvision import models

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.fc1 = nn.Linear(4096, 1600)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), (2, 2))
        # Add the first layer of convolution operation(filter size:5*5*32)followed by the first max pooling layer of size 2*2
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        # Add the second layer of convolution operation (filter sizeÅĄÅ§3*3*64) followed by the second max pooling layer of size 2*2
        x = x.view(-1, self.num_flat_features(x))
        # flatten => x shape (128, 4096) = (batch size, flatten features)
        x = self.fc1(x)  # Add a full connection layer with 1600 neurons
        x = F.dropout(x, training=self.training)  # dropout

        return x  # x shape => (128, 1600)

    def num_flat_features(self, x):  # flatten function
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class LSTM(nn.Module):
    def __init__(self, rep_dim):
        super(LSTM, self).__init__()

        self.rep_dim = rep_dim

        self.lstm1 = nn.LSTM(input_size=160, hidden_size=256, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(2560, self.rep_dim)
        # the number of neurons in the fully connected layer is equal to the number of classes of flow.

    def forward(self, x):  # x shape => 128, 1600
        # LSTM input shape => (batch,seq_len,features)
        x = x.reshape(-1, 10, 160)
        x, states = self.lstm1(x)  # Add the first, second lstm cell with 256 neurons
        # x shape => (128,10,256)
        # x = x[:, -1, :]
        x = x.reshape(-1, 2560)  # flatten => x shape (128, 2560) = (batch size, flatten features)
        x = self.fc1(x)  # Add a dense layer whose output are spatial & temporal features
        # x shape => (128,2) == (batch size, output)
        return x


class C_LSTM(nn.Module):
    def __init__(self, rep_dim):
        super(C_LSTM, self).__init__()
        self.lenet = LeNet()
        self.lstm = LSTM(rep_dim)

    def forward(self, x):
        x = x / 255. - 0.5

        # print('data.shape:',data.shape)
        # pytorch cnnCov2D input shape : batch_size, channel, height, width
        x = x.unsqueeze(1)  # setting CNN channel (1)
        x = self.lenet(x)  # pass the lenet
        x = self.lstm(x)  # pass the lstm, input shape => (128, 1600)
        return x