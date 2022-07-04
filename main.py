import argparse

import torch
from omegaconf import OmegaConf

from app import create_app

if __name__ == "__main__":
    print("===[Start]===")
    parser = argparse.ArgumentParser("Covid-EMR training script", add_help=False)
    parser.add_argument(
        "--cfg", type=str, required=True, metavar="FILE", help="path to config file"
    )
    parser.add_argument(
        "--cuda", type=str, required=True, metavar="CUDA NUMBER", help="gpu to train"
    )
    args = parser.parse_args()
    print(f"--------------------   {args.cfg}    --------------------")
    my_pipeline = OmegaConf.load(args.cfg)
    device = torch.device(
        f"cuda:{args.cuda}" if torch.cuda.is_available() == True else "cpu"
    )
    # device = torch.device("cpu")
    cfg = create_app(my_pipeline, device)
    print("===[End]===")
