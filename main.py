from omegaconf import OmegaConf

import app.models as models
from app import create_app

if __name__ == "__main__":
    print("===[Start]===")
    my_pipeline = OmegaConf.load("configs/ml_hm_los_kf10.yaml")
    cfg = create_app(my_pipeline)
    print("===[End]===")
