import pickle

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from omegaconf import OmegaConf
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


class Dataset(data.Dataset):
    def __init__(self, x, y, x_lab_length):
        self.x = x
        self.y = y
        self.x_lab_length = x_lab_length

    def __getitem__(self, index):  # 返回的是tensor
        return self.x[index], self.y[index], self.x_lab_length[index]

    def __len__(self):
        return len(self.y)
