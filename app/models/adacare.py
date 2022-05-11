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


class Sparsemax(nn.Module):
    """Sparsemax function."""

    def __init__(self, dim=None):
        super(Sparsemax, self).__init__()

        self.dim = -1 if dim is None else dim

    def forward(self, input, device="cuda"):
        original_size = input.size()
        input = input.view(-1, input.size(self.dim))

        dim = 1
        number_of_logits = input.size(dim)

        input = input - torch.max(input, dim=dim, keepdim=True)[0].expand_as(input)

        zs = torch.sort(input=input, dim=dim, descending=True)[0]
        range = torch.arange(
            start=1, end=number_of_logits + 1, device=device, dtype=torch.float32
        ).view(1, -1)
        range = range.expand_as(zs)

        bound = 1 + range * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(input.type())
        k = torch.max(is_gt * range, dim, keepdim=True)[0]

        zs_sparse = is_gt * zs
        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.expand_as(input)

        self.output = torch.max(torch.zeros_like(input), input - taus)

        output = self.output.view(original_size)

        return output

    def backward(self, grad_output):
        dim = 1

        nonzeros = torch.ne(self.output, 0)
        sum = torch.sum(grad_output * nonzeros, dim=dim) / torch.sum(nonzeros, dim=dim)
        self.grad_input = nonzeros * (grad_output - sum.expand_as(grad_output))

        return self.grad_input


class CausalConv1d(torch.nn.Conv1d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
    ):
        self.__padding = (kernel_size - 1) * dilation

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        if self.__padding != 0:
            return result[:, :, : -self.__padding]
        return result


class Recalibration(nn.Module):
    def __init__(
        self, channel, reduction=9, use_h=True, use_c=True, activation="sigmoid"
    ):
        super(Recalibration, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.use_h = use_h
        self.use_c = use_c
        scale_dim = 0
        self.activation = activation

        self.nn_c = nn.Linear(channel, channel // reduction)
        scale_dim += channel // reduction

        self.nn_rescale = nn.Linear(scale_dim, channel)
        self.sparsemax = Sparsemax(dim=1)

    def forward(self, x, device="cuda"):
        b, c, t = x.size()

        y_origin = x[:, :, -1]
        se_c = self.nn_c(y_origin)
        se_c = torch.relu(se_c)
        y = se_c

        y = self.nn_rescale(y).view(b, c, 1)
        if self.activation == "sigmoid":
            y = torch.sigmoid(y)
        else:
            y = self.sparsemax(y, device)
        return x * y.expand_as(x), y


class AdaCare(nn.Module):
    def __init__(
        self,
        hidden_dim=128,
        kernel_size=2,
        kernel_num=64,
        input_dim=76,
        output_dim=1,
        dropout=0.5,
        r_v=4,
        r_c=4,
        activation="sigmoid",
        device="cuda",
    ):
        super(AdaCare, self).__init__()
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout

        self.nn_conv1 = CausalConv1d(input_dim, kernel_num, kernel_size, 1, 1)
        self.nn_conv3 = CausalConv1d(input_dim, kernel_num, kernel_size, 1, 3)
        self.nn_conv5 = CausalConv1d(input_dim, kernel_num, kernel_size, 1, 5)
        torch.nn.init.xavier_uniform_(self.nn_conv1.weight)
        torch.nn.init.xavier_uniform_(self.nn_conv3.weight)
        torch.nn.init.xavier_uniform_(self.nn_conv5.weight)

        self.nn_convse = Recalibration(
            3 * kernel_num, r_c, use_h=False, use_c=True, activation="sigmoid"
        )
        self.nn_inputse = Recalibration(
            input_dim, r_v, use_h=False, use_c=True, activation=activation
        )
        self.rnn = nn.GRUCell(input_dim + 3 * kernel_num, hidden_dim)
        self.nn_output = nn.Linear(hidden_dim, output_dim)
        self.nn_dropout = nn.Dropout(dropout)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, input, device):
        # input shape [batch_size, timestep, feature_dim]
        batch_size = input.size(0)
        time_step = input.size(1)
        feature_dim = input.size(2)

        cur_h = Variable(torch.zeros(batch_size, self.hidden_dim)).to(device)
        inputse_att = []
        convse_att = []
        h = []

        conv_input = input.permute(0, 2, 1)
        conv_res1 = self.nn_conv1(conv_input)
        conv_res3 = self.nn_conv3(conv_input)
        conv_res5 = self.nn_conv5(conv_input)

        conv_res = torch.cat((conv_res1, conv_res3, conv_res5), dim=1)
        conv_res = self.relu(conv_res)

        for cur_time in range(time_step):
            convse_res, cur_convatt = self.nn_convse(
                conv_res[:, :, : cur_time + 1], device=device
            )
            inputse_res, cur_inputatt = self.nn_inputse(
                input[:, : cur_time + 1, :].permute(0, 2, 1), device=device
            )
            cur_input = torch.cat((convse_res[:, :, -1], inputse_res[:, :, -1]), dim=-1)

            cur_h = self.rnn(cur_input, cur_h)
            h.append(cur_h)
            convse_att.append(cur_convatt)
            inputse_att.append(cur_inputatt)

        h = torch.stack(h).permute(1, 0, 2)
        h_reshape = h.contiguous().view(batch_size * time_step, self.hidden_dim)
        if self.dropout > 0.0:
            h_reshape = self.nn_dropout(h_reshape)
        output = self.nn_output(h_reshape)
        output = self.sigmoid(output)
        output = output.contiguous().view(batch_size, time_step, self.output_dim)
        return output, inputse_att
