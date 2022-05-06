import numpy as np
import argparse
import os
import re
import pickle
import datetime
import random
import math
import copy

from sklearn.model_selection import KFold, StratifiedKFold
import torch
from torch import nn
import torch.nn.utils.rnn as rnn_utils
from torch.utils import data
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset, Subset

class GRU(nn.Module):
    def __init__(self, input_lab_dim, input_demo_dim, hidden_dim, output_dim, act_layer=nn.GELU, drop=0.):
        super(GRU, self).__init__()

        # hyperparameters
        self.input_lab_dim = input_lab_dim
        self.input_demo_dim = input_demo_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.demo_proj = nn.Linear(input_demo_dim, hidden_dim)
        self.lab_proj = nn.Linear(input_lab_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(13)

        self.gru = nn.GRU(input_size = hidden_dim, hidden_size = hidden_dim, num_layers = 1, batch_first = True)
        
        self.act = act_layer()
        self.fc = nn.Linear(2*hidden_dim, output_dim)
        self.drop = nn.Dropout(drop)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x_lab, x_lab_length, x_demo):
        batch_size, max_length, input_dim = x_lab.shape
        x_lab = self.lab_proj(x_lab)
        
        # x_lab = self.act(x_lab)
        # x_lab = self.bn(x_lab)

        # x_lab = torch.nn.utils.rnn.pack_padded_sequence(x_lab, x_lab_length, batch_first=True, enforce_sorted=False)
        x_lab, h_n = self.gru(x_lab) # output: (batch_size,L,hidden_dim) h_n: (1, batch_size, hidden_dim)

        x_demo = self.demo_proj(x_demo)

        x_demo = torch.reshape(x_demo.repeat(1, max_length), (batch_size, max_length, self.hidden_dim))

        x = torch.cat((x_lab, x_demo), 2) # (batch_size, 2*hidden_dim)

        x = self.drop(x)
        x = self.fc(x)
        x = self.drop(x)
        
        x = self.sigmoid(x)
        return x