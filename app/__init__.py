from omegaconf import OmegaConf

from app import core, dataset, models, utils


def create_app(my_pipeline):
    # Load dataset
    dataset_cfg = OmegaConf.load(f"configs/_base_/dataset/{my_pipeline.dataset}.yaml")
    # Merge config
    cfg = OmegaConf.merge(dataset_cfg, my_pipeline)

    print(cfg.model_type, cfg.model, cfg.task)
    if cfg.model_type == "ml" and cfg.task == "los":
        core.ml_los_pipeline.start_pipeline(cfg)

    return cfg
