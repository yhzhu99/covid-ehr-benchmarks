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

def create_app(my_pipeline):
    # Load config
    my_pipeline = OmegaConf.load('configs/gru_tongji_epoch50_fold10_bs64.yaml')
    # Load dataset
    dataset = OmegaConf.load(f'configs/_base_/dataset/{my_pipeline.dataset}.yaml')
    # Merge config
    cfg = OmegaConf.merge(dataset, my_pipeline)
    # Create model
    model = eval(f'models.{cfg.model}(input_lab_dim={cfg.labtest_dim}, input_demo_dim={cfg.demographic_dim}, hidden_dim={32}, output_dim={1})')
    # Print model
    return cfg