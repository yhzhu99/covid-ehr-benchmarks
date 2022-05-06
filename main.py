from omegaconf import OmegaConf

import app.models as models
from app import create_app

if __name__ == '__main__':
    my_pipeline = OmegaConf.load('configs/gru_tongji_epoch50_fold10_bs64.yaml')
    cfg = create_app(my_pipeline)
    print(cfg)