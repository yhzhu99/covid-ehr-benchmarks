import argparse

import torch
from omegaconf import OmegaConf

import app.models as models
from app import create_app

if __name__ == "__main__":
    print("===[Start]===")

    parser = argparse.ArgumentParser("Covid-EMR training script", add_help=False)
    parser.add_argument(
        "--cfg", type=str, required=True, metavar="FILE", help="path to config file"
    )
    args = parser.parse_args()
    my_pipeline = OmegaConf.load(args.cfg)
    device = torch.device("cuda:0" if torch.cuda.is_available() == True else "cpu")
    cfg = create_app(my_pipeline, device)
    print("===[End]===")
