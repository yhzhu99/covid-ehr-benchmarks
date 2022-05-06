import numpy as np
import argparse
import os
import re
import pickle
import datetime
import random
import math
import copy

from omegaconf import OmegaConf

from sklearn.model_selection import KFold, StratifiedKFold
import torch
from torch import nn
import torch.nn.utils.rnn as rnn_utils
from torch.utils import data
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset, Subset

import app.models as models
from app import create_app

if __name__ == '__main__':
    my_pipeline = OmegaConf.load('configs/gru_tongji_epoch50_fold10_bs64.yaml')
    cfg = create_app(my_pipeline)
    print(cfg)