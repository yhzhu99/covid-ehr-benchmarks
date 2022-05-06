from omegaconf import OmegaConf

import app.models as models

def create_app(my_pipeline):
    # Load config
    default_pipeline = OmegaConf.load('configs/gru_tongji_epoch50_fold10_bs64.yaml')
    # Load dataset
    dataset = OmegaConf.load(f'configs/_base_/dataset/{default_pipeline.dataset}.yaml')
    # Merge config
    cfg = OmegaConf.merge(dataset, default_pipeline, my_pipeline)
    # Create model
    model = eval(f'models.{cfg.model}(input_lab_dim={cfg.labtest_dim}, input_demo_dim={cfg.demographic_dim}, hidden_dim={32}, output_dim={1})')
    # Print model
    return cfg