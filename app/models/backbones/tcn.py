import argparse
import copy
import datetime
import math
import os
import pickle
import random
import re

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from sklearn.model_selection import KFold, StratifiedKFold
from torch import nn
from torch.autograd import Variable
from torch.nn.utils import weight_norm
from torch.utils import data
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    Subset,
    SubsetRandomSampler,
    TensorDataset,
    random_split,
)


# From TCN original paper https://github.com/locuslab/TCN
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2
    ):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            ),
            dim=None,
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            ),
            dim=None,
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


# From TCN original paper https://github.com/locuslab/TCN
class TemporalConvNet(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_channels,  # serve as hidden dim
        max_seq_length=0,
        kernel_size=2,
        dropout=0.0,
    ):
        super(TemporalConvNet, self).__init__()
        self.num_channels = num_channels

        layers = []

        # We compute automatically the depth based on the desired seq_length.
        if isinstance(num_channels, int) and max_seq_length:
            num_channels = [num_channels] * int(
                np.ceil(np.log(max_seq_length / 2) / np.log(kernel_size))
            )
        elif isinstance(num_channels, int) and not max_seq_length:
            raise Exception(
                "a maximum sequence length needs to be provided if num_channels is int"
            )

        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]

        self.network = nn.Sequential(*layers)

    def forward(self, x, device, info=None):
        """extra info is not used here"""
        batch_size, time_steps, _ = x.size()
        out = torch.zeros((batch_size, time_steps, self.num_channels)).to(device)
        for cur_time in range(time_steps):
            cur_x = x[:, : cur_time + 1, :]
            cur_x = cur_x.permute(0, 2, 1)  # Permute to channel first
            o = self.network(cur_x)
            o = o.permute(0, 2, 1)  # Permute to channel last
            out[:, cur_time, :] = torch.mean(o, dim=1)
        return out
