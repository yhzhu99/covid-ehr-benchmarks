import json
import time

from omegaconf import OmegaConf
from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, relationship, sessionmaker

db_cfg = OmegaConf.load("configs/_base_/db.yaml")

username, password, host, port, database = (
    db_cfg.username,
    db_cfg.password,
    db_cfg.host,
    db_cfg.port,
    db_cfg.database,
)
SQLALCHEMY_DATABASE_URL = f"mysql://{username}:{password}@{host}:{port}/{database}"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Perflog(Base):
    __tablename__ = "perflog"
    id = Column(Integer, primary_key=True, index=True)
    task = Column(String)
    model_type = Column(String)
    model_name = Column(String)
    performance = Column(String)
    config = Column(String)
    record_time = Column(Integer)


def process_performance_raw_info(
    cfg,
    mae=None,
    mse=None,
    rmse=None,
    mape=None,
    acc=None,
    auroc=None,
    auprc=None,
    early_prediction_score=None,
    multitask_prediction_score=None,
    verbose=0,
):
    result = []
    if mae is not None:
        result.extend(
            [
                {"name": "mae", "mean": mae.mean(), "std": mae.std()},
                {"name": "mse", "mean": mse.mean(), "std": mse.std()},
                {"name": "rmse", "mean": rmse.mean(), "std": rmse.std()},
                {"name": "mape", "mean": mape.mean(), "std": mape.std()},
            ]
        )
    if acc is not None:
        result.extend(
            [
                {"name": "acc", "mean": acc.mean(), "std": acc.std()},
                {"name": "auroc", "mean": auroc.mean(), "std": auroc.std()},
                {"name": "auprc", "mean": auprc.mean(), "std": auprc.std()},
            ]
        )
    thresholds = cfg.thresholds
    if early_prediction_score is not None:
        for i in range(len(thresholds)):
            result.append(
                {
                    "name": "early_prediction_score",
                    "mean": early_prediction_score.mean(axis=0)[i],
                    "std": early_prediction_score.std(axis=0)[i],
                    "threshold": thresholds[i],
                }
            )
    if multitask_prediction_score is not None:
        for i in range(len(thresholds)):
            result.append(
                {
                    "name": "multitask_prediction_score",
                    "mean": multitask_prediction_score.mean(axis=0)[i],
                    "std": multitask_prediction_score.std(axis=0)[i],
                    "threshold": thresholds[i],
                }
            )
    if verbose == 1:
        print(result)
    return result


def create_perflog(db: Session, cfg, perf=None):
    db_perflog = Perflog(
        task=cfg.task,
        model_type=cfg.model_type,
        model_name=cfg.model,
        performance=json.dumps(perf),
        config=OmegaConf.to_yaml(cfg),
        record_time=int(time.time()),
    )
    db.add(db_perflog)
    db.commit()
    db.refresh(db_perflog)
    return db_perflog


def process_and_upload_performance(
    cfg,
    mae=None,
    mse=None,
    rmse=None,
    mape=None,
    acc=None,
    auroc=None,
    auprc=None,
    early_prediction_score=None,
    multitask_prediction_score=None,
    verbose=0,
):
    db = SessionLocal()
    perf = process_performance_raw_info(
        cfg,
        mae=mae,
        mse=mse,
        rmse=rmse,
        mape=mape,
        acc=acc,
        auroc=auroc,
        auprc=auprc,
        early_prediction_score=early_prediction_score,
        multitask_prediction_score=multitask_prediction_score,
        verbose=verbose,
    )
    create_perflog(db=db, cfg=cfg, perf=perf)
    db.close()
