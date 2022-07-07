from omegaconf import OmegaConf

from app import apis, core, datasets, models
from app.core.utils import RANDOM_SEED, init_random

init_random()


def create_app(my_pipeline, device):
    # Load dataset
    dataset_cfg = OmegaConf.load(f"configs/_base_/datasets/{my_pipeline.dataset}.yaml")
    # Merge config
    cfg = OmegaConf.merge(dataset_cfg, my_pipeline)

    print(cfg.model_type, cfg.model, cfg.task)
    if cfg.model_type == "ml" and cfg.task == "los":
        apis.ml_los_pipeline.start_pipeline(cfg)
    elif cfg.model_type == "ml" and cfg.task == "outcome":
        apis.ml_outcome_pipeline.start_pipeline(cfg)
    elif cfg.model_type in ["dl", "ehr"] and cfg.task == "los":
        apis.dl_los_pipeline.start_pipeline(cfg, device)
    elif cfg.model_type in ["dl", "ehr"] and cfg.task == "outcome":
        apis.dl_outcome_pipeline.start_pipeline(cfg, device)
    elif cfg.model_type in ["dl", "ehr"] and cfg.task == "multitask":
        apis.dl_multitask_pipeline.start_pipeline(cfg, device)
        # apis.dl_multitask_pipeline.start_inference(cfg, device)
    elif cfg.task == "twostage":
        apis.dl_twostage_pipeline.start_pipeline(cfg, device)
