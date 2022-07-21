import math
import pathlib
import pickle
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from omegaconf import OmegaConf
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
    train_test_split,
)
from sklearn.tree import DecisionTreeRegressor
from torch import nn
from torch.autograd import Variable
from torch.utils import data
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    Subset,
    SubsetRandomSampler,
    TensorDataset,
    random_split,
)

from app.core.evaluation import covid_metrics, eval_metrics
from app.core.utils import RANDOM_SEED
from app.datasets import get_dataset, load_data
from app.datasets.dl import Dataset
from app.datasets.ml import flatten_dataset, numpy_dataset
from app.models import (
    build_model_from_cfg,
    get_multi_task_loss,
    predict_all_visits_bce_loss,
    predict_all_visits_mse_loss,
)
from app.utils import perflog


def twostage_inference(
    outcome_model, los_model, device, data, loss_fn, los_statistics, info
):
    """
    val / test
    """
    val_loss = []
    y_outcome_pred = []
    y_outcome_true = []
    y_los_pred = []
    y_los_true = []
    y_true_all = []
    with torch.no_grad():
        batch_x, batch_y, batch_x_lab_length = data
        batch_x, batch_y, batch_x_lab_length = (
            batch_x.float().to(device),
            batch_y.float().to(device),
            batch_x_lab_length.float().to(device),
        )
        all_y = batch_y
        batch_y_outcome = batch_y[:, :, 0].unsqueeze(-1)
        batch_y_los = batch_y[:, :, 1].unsqueeze(-1)
        outcome = outcome_model(batch_x, device, info)
        los = los_model(batch_x, device, info)
        loss = loss_fn(outcome, batch_y_outcome, los, batch_y_los, batch_x_lab_length)
        val_loss.append(loss.item())
        los = torch.squeeze(los)
        batch_y_los = torch.squeeze(batch_y_los)
        for i in range(len(batch_y_outcome)):
            y_outcome_pred.extend(outcome[i][: batch_x_lab_length[i].long()].tolist())
            y_outcome_true.extend(
                batch_y_outcome[i][: batch_x_lab_length[i].long()].tolist()
            )
            y_los_pred.extend(los[i][: batch_x_lab_length[i].long()].tolist())
            y_los_true.extend(batch_y_los[i][: batch_x_lab_length[i].long()].tolist())
            y_true_all.extend(all_y[i][: batch_x_lab_length[i].long()].tolist())
    y_outcome_true = np.array(y_outcome_true)
    y_outcome_pred = np.array(y_outcome_pred)
    y_true_all = np.array(y_true_all)
    y_los_true = np.array(y_los_true)
    y_los_pred = np.array(y_los_pred)
    y_true_all = reverse_zscore_los(y_true_all, los_statistics)
    y_los_true = reverse_zscore_los(y_los_true, los_statistics)
    y_los_pred = reverse_zscore_los(y_los_pred, los_statistics)
    early_prediction_score = covid_metrics.early_prediction_outcome_metric(
        y_true_all, y_outcome_pred, info["config"].thresholds, verbose=0
    )
    multitask_los_score = covid_metrics.multitask_los_metric(
        y_true_all,
        y_outcome_pred,
        y_los_pred,
        info["config"].large_los,
        info["config"].thresholds,
        verbose=0,
    )
    y_outcome_pred = np.stack([1 - y_outcome_pred, y_outcome_pred], axis=1)
    outcome_evaluation_scores = eval_metrics.print_metrics_binary(
        y_outcome_true, y_outcome_pred, verbose=0
    )
    los_evaluation_scores = eval_metrics.print_metrics_regression(
        y_los_true, y_los_pred, verbose=0
    )
    covid_evaluation_scores = {
        "early_prediction_score": early_prediction_score,
        "multitask_los_score": multitask_los_score,
    }
    return (
        np.array(val_loss).mean(),
        outcome_evaluation_scores,
        los_evaluation_scores,
        covid_evaluation_scores,
    )


def calculate_los_statistics(dataset, train_idx):
    """calculate los's mean/std"""
    y = []
    for i in train_idx:
        for j in range(dataset.x_lab_length[i]):
            y.append(dataset.y[i][j][1])
    y = np.array(y)
    mean, std = y.mean(), y.std()
    los_statistics = {"los_mean": mean, "los_std": std}
    return los_statistics


def zscore_los(dataset, los_statistics):
    """zscore scale y"""
    dataset.y[:, :, 1] = (
        dataset.y[:, :, 1] - los_statistics["los_mean"]
    ) / los_statistics["los_std"]
    return dataset


def reverse_zscore_los(y, los_statistics):
    """reverse zscore y"""
    if len(y.shape) == 1:
        y = y * los_statistics["los_std"] + los_statistics["los_mean"]
    elif len(y.shape) == 2:  # [outcome, los]
        y[:, 1] = y[:, 1] * los_statistics["los_std"] + los_statistics["los_mean"]
    return y


def start_pipeline(cfg, device):
    val_info = {"config": cfg, "epoch": cfg.epochs}
    dataset_type, method, num_folds, train_fold = (
        cfg.dataset,
        cfg.model,
        cfg.num_folds,
        cfg.train_fold,
    )
    # Load data
    x, y, x_lab_length = load_data(dataset_type)
    dataset = get_dataset(x, y, x_lab_length)

    # Load dataset
    dataset_cfg = OmegaConf.load(f"configs/_base_/datasets/{cfg.dataset}.yaml")

    # Merge config
    outcome_cfg = OmegaConf.merge(
        dataset_cfg, OmegaConf.load(f"configs/{cfg.outcome_model_name}.yaml")
    )
    outcome_model = build_model_from_cfg(outcome_cfg, device)

    # Merge config
    los_cfg = OmegaConf.merge(
        dataset_cfg, OmegaConf.load(f"configs/{cfg.los_model_name}.yaml")
    )
    los_model = build_model_from_cfg(los_cfg, device)

    criterion = get_multi_task_loss

    all_history = {}
    test_performance = {
        "test_loss": [],
        "test_mad": [],
        "test_mse": [],
        "test_mape": [],
        "test_rmse": [],
        "test_accuracy": [],
        "test_auroc": [],
        "test_auprc": [],
        "test_early_prediction_score": [],
        "test_multitask_los_score": [],
    }
    kfold_test = StratifiedKFold(
        n_splits=num_folds, shuffle=True, random_state=RANDOM_SEED
    )
    if cfg.bootstrap == True:
        kfold_test = StratifiedShuffleSplit(
            n_splits=num_folds, test_size=1 / num_folds, random_state=RANDOM_SEED
        )
    skf = kfold_test.split(np.arange(len(dataset)), dataset.y[:, 0, 0])
    for fold_test in range(train_fold):
        outcome_model.load_state_dict(
            torch.load(f"checkpoints/{cfg.outcome_model_name}.pth")
        )
        los_model.load_state_dict(torch.load(f"checkpoints/{cfg.los_model_name}.pth"))
        x, y, x_lab_length = load_data(dataset_type)
        dataset = get_dataset(x, y, x_lab_length)
        train_and_val_idx, test_idx = next(skf)
        print("====== Test Fold {} ======".format(fold_test + 1))
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=1 / (num_folds - 1), random_state=RANDOM_SEED
        )

        sub_dataset = Dataset(
            dataset.x[train_and_val_idx],
            dataset.y[train_and_val_idx],
            dataset.x_lab_length[train_and_val_idx],
        )
        all_history["test_fold_{}".format(fold_test + 1)] = {}

        train_idx, val_idx = next(
            sss.split(np.arange(len(train_and_val_idx)), sub_dataset.y[:, 0, 0])
        )

        # apply z-score transform los
        los_statistics = calculate_los_statistics(sub_dataset, train_idx)
        print(los_statistics)
        sub_dataset = zscore_los(sub_dataset, los_statistics)
        dataset = zscore_los(dataset, los_statistics)

        (
            test_loss,
            test_outcome_evaluation_scores,
            test_los_evaluation_scores,
            test_covid_evaluation_scores,
        ) = twostage_inference(
            outcome_model,
            los_model,
            device,
            dataset[test_idx],
            criterion,
            los_statistics,
            info=val_info,
        )

        test_performance["test_loss"].append(test_loss)
        test_performance["test_mad"].append(test_los_evaluation_scores["mad"])
        test_performance["test_mse"].append(test_los_evaluation_scores["mse"])
        test_performance["test_mape"].append(test_los_evaluation_scores["mape"])
        test_performance["test_rmse"].append(test_los_evaluation_scores["rmse"])
        test_performance["test_accuracy"].append(test_outcome_evaluation_scores["acc"])
        test_performance["test_auroc"].append(test_outcome_evaluation_scores["auroc"])
        test_performance["test_auprc"].append(test_outcome_evaluation_scores["auprc"])
        test_performance["test_early_prediction_score"].append(
            test_covid_evaluation_scores["early_prediction_score"]
        )
        test_performance["test_multitask_los_score"].append(
            test_covid_evaluation_scores["multitask_los_score"]
        )
        print(
            f"Performance on test set {fold_test+1}: MAE = {test_los_evaluation_scores['mad']}, MSE = {test_los_evaluation_scores['mse']}, RMSE = {test_los_evaluation_scores['rmse']}, MAPE = {test_los_evaluation_scores['mape']}, ACC = {test_outcome_evaluation_scores['acc']}, AUROC = {test_outcome_evaluation_scores['auroc']}, AUPRC = {test_outcome_evaluation_scores['auprc']},  EarlyPredictionScore = {test_covid_evaluation_scores['early_prediction_score']}, MultitaskPredictionScore = {test_covid_evaluation_scores['multitask_los_score']}"
        )
    # Calculate average performance on 10-fold test set
    test_mad_list = np.array(test_performance["test_mad"])
    test_mse_list = np.array(test_performance["test_mse"])
    test_mape_list = np.array(test_performance["test_mape"])
    test_rmse_list = np.array(test_performance["test_rmse"])
    test_accuracy_list = np.array(test_performance["test_accuracy"])
    test_auroc_list = np.array(test_performance["test_auroc"])
    test_auprc_list = np.array(test_performance["test_auprc"])
    test_early_prediction_list = np.array(
        test_performance["test_early_prediction_score"]
    )
    test_multitask_los_list = np.array(test_performance["test_multitask_los_score"])

    print("====================== TEST RESULT ======================")
    print("MAE: {:.3f} ({:.3f})".format(test_mad_list.mean(), test_mad_list.std()))
    print("MSE: {:.3f} ({:.3f})".format(test_mse_list.mean(), test_mse_list.std()))
    print("MAPE: {:.3f} ({:.3f})".format(test_mape_list.mean(), test_mape_list.std()))
    print("RMSE: {:.3f} ({:.3f})".format(test_rmse_list.mean(), test_rmse_list.std()))
    print(
        "ACC: {:.3f} ({:.3f})".format(
            test_accuracy_list.mean(), test_accuracy_list.std()
        )
    )
    print(
        "AUROC: {:.3f} ({:.3f})".format(test_auroc_list.mean(), test_auroc_list.std())
    )
    print(
        "AUPRC: {:.3f} ({:.3f})".format(test_auprc_list.mean(), test_auprc_list.std())
    )
    print(
        "EarlyPredictionScore:",
        (
            test_early_prediction_list.mean(axis=0),
            test_early_prediction_list.std(axis=0),
        ),
    )

    for i in range(len(cfg.thresholds)):
        print(
            cfg.thresholds[i],
            test_early_prediction_list.mean(axis=0)[i],
            test_early_prediction_list.std(axis=0)[i],
        )

    print(
        "MultitaskPredictionScore:",
        (test_multitask_los_list.mean(axis=0), test_multitask_los_list.std(axis=0)),
    )

    for i in range(len(cfg.thresholds)):
        print(
            cfg.thresholds[i],
            test_multitask_los_list.mean(axis=0)[i],
            test_multitask_los_list.std(axis=0)[i],
        )
    print("=========================================================")
    perflog.process_and_upload_performance(
        cfg,
        mae=test_mad_list,
        mse=test_mse_list,
        rmse=test_rmse_list,
        mape=test_mape_list,
        acc=test_accuracy_list,
        auroc=test_auroc_list,
        auprc=test_auprc_list,
        early_prediction_score=test_early_prediction_list,
        multitask_prediction_score=test_multitask_los_list,
        verbose=1,
        upload=cfg.db,
    )
