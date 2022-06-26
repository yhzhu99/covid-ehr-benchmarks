import torch
from omegaconf import OmegaConf

import app.models as models
from app import create_app

if __name__ == "__main__":
    print("===[Start]===")
    my_pipeline = OmegaConf.load("configs/tj_multitask_gru_ep100_kf10_bs64_hid64.yaml")
    device = torch.device("cuda:0" if torch.cuda.is_available() == True else "cpu")
    cfg = create_app(my_pipeline, device)
    print("===[End]===")
