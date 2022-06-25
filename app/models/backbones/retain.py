import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RETAIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super(RETAIN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.gru_a = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.gru_b = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.a_mat = nn.Linear(hidden_dim, 1)
        self.b_mat = nn.Linear(hidden_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)

    def retain_encoder(self, x):
        x_a, _ = self.gru_a(x)
        alpha = self.softmax(self.a_mat(x_a))
        x_b, _ = self.gru_b(x)
        beta = torch.tanh(self.b_mat(x_b))
        out = ((x * beta).transpose(1, 2) @ alpha).squeeze(-1)
        return out

    def forward(self, x):
        batch_size, time_steps, _ = x.size()
        x = self.proj(x)
        x = self.dropout(x)

        out = torch.zeros((batch_size, time_steps, self.hidden_dim))

        for cur_time in range(time_steps):
            cur_x = x[:, : cur_time + 1, :]
            out[:, cur_time, :] = self.retain_encoder(cur_x)
        return out
