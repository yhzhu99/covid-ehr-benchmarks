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

from app import models


class Dataset(data.Dataset):
    def __init__(self, x, y, x_lab_length):
        self.x = x
        self.y = y
        self.x_lab_length = x_lab_length

    def __getitem__(self, index):  # 返回的是tensor
        return self.x[index], self.y[index], self.x_lab_length[index]

    def __len__(self):
        return len(self.y)


def load_data(dataset_type):
    # Load data
    data_path = f"../dataset/{dataset_type}/processed_data/"
    x = pickle.load(open(data_path + "x.pkl", "rb"))
    y = pickle.load(open(data_path + "y.pkl", "rb"))
    x_lab_length = pickle.load(open(data_path + "visits_length.pkl", "rb"))

    return x, y, x_lab_length


def create_app(my_pipeline):
    # Load config
    my_pipeline = OmegaConf.load("configs/gru_tongji_epoch50_fold10_bs64.yaml")
    # Load dataset
    dataset = OmegaConf.load(f"configs/_base_/dataset/{my_pipeline.dataset}.yaml")
    # Merge config
    cfg = OmegaConf.merge(dataset, my_pipeline)
    return cfg
