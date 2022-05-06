from omegaconf import OmegaConf
import pickle
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
import torch
from torch import nn
import torch.nn.utils.rnn as rnn_utils
from torch.utils import data
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset, Subset

import app.models as models

class Dataset(data.Dataset):
    def __init__(self, x_lab, x_lab_length, x_demo, y_outcome, y_los):
        self.x_lab = x_lab
        self.x_lab_length = x_lab_length
        self.x_demo = x_demo
        self.y_outcome = y_outcome
        self.y_los = y_los

    def __getitem__(self, index): # 返回的是tensor
        return self.x_lab[index], self.x_lab_length[index], self.x_demo[index], self.y_outcome[index], self.y_los[index]

    def __len__(self):
        return len(self.y_outcome)

def load_data():
    # Load data
    x_lab = pickle.load(open('./dataset/tongji/processed_data/train_x_labtest.pkl', 'rb'))
    x_lab = np.array(x_lab, dtype=object)
    x_lab = [torch.Tensor(_) for _ in x_lab]

    x_demo = pickle.load(open('./dataset/tongji/processed_data/train_x_demographic.pkl', 'rb'))
    x_demo = np.array(x_demo)

    y_outcome = pickle.load(open('./dataset/tongji/processed_data/train_y_outcome.pkl', 'rb'))
    y_outcome = np.array(y_outcome)

    y_los = pickle.load(open('./dataset/tongji/processed_data/train_y_LOS.pkl', 'rb'))
    y_los = np.array(y_los, dtype=object)
    y_los = [torch.Tensor(_) for _ in y_los]

    x_lab_length = [len(_) for _ in x_lab]
    x_lab_length = np.array(x_lab_length)
    x_lab = torch.nn.utils.rnn.pad_sequence((x_lab), batch_first=True)
    y_los = torch.nn.utils.rnn.pad_sequence(y_los, batch_first=True)

    train_dataset = Dataset(x_lab, x_lab_length, x_demo, y_outcome, y_los)

    return train_dataset

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

    train_dataset = load_data()
    dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size)

    model.train()
    for step, data in enumerate(dataloader):   
        batch_x_lab, batch_x_lab_length, batch_x_demo, batch_y_outcome, batch_y_los = data
        batch_x_lab, batch_x_lab_length, batch_x_demo, batch_y_outcome, batch_y_los = batch_x_lab.float(),batch_x_lab_length.float() , batch_x_demo.float(), batch_y_outcome.float(), batch_y_los.float()
        batch_y_outcome = batch_y_outcome.unsqueeze(-1)
        batch_y_los = batch_y_los.unsqueeze(-1)
        outcome = model(batch_x_lab, batch_x_lab_length, batch_x_demo)

    print(outcome.shape)

    return cfg