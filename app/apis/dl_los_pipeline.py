import math
import pathlib
import pickle
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
    train_test_split,
)
from sklearn.tree import DecisionTreeRegressor
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

from app import datasets
from app.datasets.ml import flatten_dataset, numpy_dataset
from app.utils import RANDOM_SEED, metrics


def train_epoch(model, device, dataloader, loss_fn, optimizer):
    train_loss = []
    model.train()
    for step, data in enumerate(dataloader):
        batch_x, batch_y, batch_x_lab_length = data
        batch_x, batch_y, batch_x_lab_length = (
            batch_x.float(),
            batch_y.float(),
            batch_x_lab_length.float(),
        )
        batch_y = batch_y[:, :, 0]  # 0: outcome, 1: los
        batch_y = batch_y.unsqueeze(-1)
        optimizer.zero_grad()
        output = model(batch_x)
        loss = loss_fn(output, batch_y, batch_x_lab_length)
        train_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    return np.array(train_loss).mean()


def val_epoch(model, device, dataloader, loss_fn):
    """
    val / test
    """
    val_loss = []
    y_pred = []
    y_true = []
    model.eval()
    with torch.no_grad():
        for step, data in enumerate(dataloader):
            batch_x, batch_y, batch_x_lab_length = data
            batch_x, batch_y, batch_x_lab_length = (
                batch_x.float(),
                batch_y.float(),
                batch_x_lab_length.float(),
            )
            batch_y = batch_y[:, :, 0]  # 0: outcome, 1: los
            batch_y = batch_y.unsqueeze(-1)
            output = model(batch_x)
            loss = loss_fn(output, batch_y, batch_x_lab_length)
            val_loss.append(loss.item())
            for i in range(len(batch_y)):
                y_pred.extend(output[i][: batch_x_lab_length[i].long()].tolist())
                y_true.extend(batch_y[i][: batch_x_lab_length[i].long()].tolist())
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred = np.stack([1 - y_pred, y_pred], axis=1)
    evaluation_scores = metrics.print_metrics_binary(y_true, y_pred, verbose=0)
    return np.array(val_loss).mean(), evaluation_scores


def start_pipeline(cfg):
    pass
