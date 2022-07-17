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
        "--cuda",
        type=int,
        required=False,
        metavar="CUDA NUMBER",
        help="gpu to train",
    )
    parser.add_argument(
        "--db",
        type=bool,
        default=False,
    )

    args = parser.parse_args()
    print(f"===[{args.cfg}]===")
    conf = OmegaConf.load(args.cfg)
    conf.db = args.db

    # train on cpu by default
    device = torch.device("cpu")
    if args.cuda is not None:
        device = torch.device(
            f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
        )

    create_app(conf, device)
    print("===[End]===")
