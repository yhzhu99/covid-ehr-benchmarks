from omegaconf import OmegaConf

from app import core, datasets, models, utils
from app.utils import init_random

init_random()


def create_app(my_pipeline):
    # Load dataset
    dataset_cfg = OmegaConf.load(f"configs/_base_/datasets/{my_pipeline.dataset}.yaml")
    # Merge config
    cfg = OmegaConf.merge(dataset_cfg, my_pipeline)

    print(cfg.model_type, cfg.model, cfg.task)
    if cfg.model_type == "ml" and cfg.task == "los":
        core.ml_los_pipeline.start_pipeline(cfg)
    elif cfg.model_type == "ml" and cfg.task == "outcome":
        core.ml_outcome_pipeline.start_pipeline(cfg)

    return cfg
