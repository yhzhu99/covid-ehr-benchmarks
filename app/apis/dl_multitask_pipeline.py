import math
import pathlib
import pickle
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
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


def train_epoch(model, device, dataloader, loss_fn, optimizer):
    train_loss = []
    model.train()
    for step, data in enumerate(dataloader):
        batch_x, batch_y, batch_x_lab_length = data
        batch_x, batch_y, batch_x_lab_length = (
            batch_x.float(),
            batch_y.float(),
            batch_x_lab_length.float(),
        )
        batch_y_outcome = batch_y[:, :, 0].unsqueeze(-1)
        batch_y_los = batch_y[:, :, 1].unsqueeze(-1)
        optimizer.zero_grad()
        outcome, los = model(batch_x)
        loss = loss_fn(outcome, batch_y_outcome, los, batch_y_los, batch_x_lab_length)
        train_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    return np.array(train_loss).mean()


def val_epoch(model, device, dataloader, loss_fn, los_statistics):
    """
    val / test
    """
    val_loss = []
    y_outcome_pred = []
    y_outcome_true = []
    y_los_pred = []
    y_los_true = []
    y_true_all = []
    model.eval()
    with torch.no_grad():
        for step, data in enumerate(dataloader):
            batch_x, batch_y, batch_x_lab_length = data
            batch_x, batch_y, batch_x_lab_length = (
                batch_x.float(),
                batch_y.float(),
                batch_x_lab_length.float(),
            )
            all_y = batch_y
            batch_y_outcome = batch_y[:, :, 0].unsqueeze(-1)
            batch_y_los = batch_y[:, :, 1].unsqueeze(-1)
            outcome, los = model(batch_x)
            loss = loss_fn(
                outcome, batch_y_outcome, los, batch_y_los, batch_x_lab_length
            )
            val_loss.append(loss.item())
            los = torch.squeeze(los)
            batch_y_los = torch.squeeze(batch_y_los)
            for i in range(len(batch_y_outcome)):
                y_outcome_pred.extend(
                    outcome[i][: batch_x_lab_length[i].long()].tolist()
                )
                y_outcome_true.extend(
                    batch_y_outcome[i][: batch_x_lab_length[i].long()].tolist()
                )
                y_los_pred.extend(los[i][: batch_x_lab_length[i].long()].tolist())
                y_los_true.extend(
                    batch_y_los[i][: batch_x_lab_length[i].long()].tolist()
                )
                y_true_all.extend(all_y[i][: batch_x_lab_length[i].long()].tolist())
    y_outcome_true = np.array(y_outcome_true)
    y_outcome_pred = np.array(y_outcome_pred)
    y_true_all = np.array(y_true_all)
    y_los_true = np.array(y_los_true)
    y_los_pred = np.array(y_los_pred)
    early_prediction_score = covid_metrics.early_prediction_outcome_metric(
        y_true_all, y_outcome_pred, verbose=0
    )
    multitask_los_score = covid_metrics.multitask_los_metric(
        y_true_all,
        y_outcome_pred,
        y_los_pred,
        metrics_strategy="MAE",
        verbose=1,
    )
    y_outcome_pred = np.stack([1 - y_outcome_pred, y_outcome_pred], axis=1)
    outcome_evaluation_scores = eval_metrics.print_metrics_binary(
        y_outcome_true, y_outcome_pred, verbose=0
    )
    y_los_true = reverse_zscore_los(y_los_true, los_statistics)
    y_los_pred = reverse_zscore_los(y_los_pred, los_statistics)
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
    y = y * los_statistics["los_std"] + los_statistics["los_mean"]
    return y


def start_pipeline(cfg, device):
    dataset_type, method, num_folds, train_fold = (
        cfg.dataset,
        cfg.model,
        cfg.num_folds,
        cfg.train_fold,
    )
    # Load data
    x, y, x_lab_length = load_data(dataset_type)
    dataset = get_dataset(x, y, x_lab_length)
    model = build_model_from_cfg(cfg)
    print(model)
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
    skf = kfold_test.split(np.arange(len(dataset)), dataset.y[:, 0, 0])
    for fold_test in range(train_fold):
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

        test_sampler = SubsetRandomSampler(test_idx)
        test_loader = DataLoader(
            dataset, batch_size=cfg.batch_size, sampler=test_sampler
        )
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(
            dataset, batch_size=cfg.batch_size, sampler=train_sampler
        )
        val_loader = DataLoader(dataset, batch_size=cfg.batch_size, sampler=val_sampler)
        model = build_model_from_cfg(cfg)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = get_multi_task_loss
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_mad": [],
            "val_mse": [],
            "val_mape": [],
            "val_rmse": [],
            "val_accuracy": [],
            "val_auroc": [],
            "val_auprc": [],
            "val_early_prediction_score": [],
            "val_multitask_los_score": [],
        }
        best_val_performance = 1e8
        for epoch in range(cfg.epochs):
            train_loss = train_epoch(model, device, train_loader, criterion, optimizer)
            (
                val_loss,
                val_outcome_evaluation_scores,
                val_los_evaluation_scores,
                val_covid_evaluation_scores,
            ) = val_epoch(model, device, val_loader, criterion, los_statistics)
            # save performance history on validation set
            print(
                "Epoch:{}/{} AVG Training Loss:{:.3f} AVG Val Loss:{:.3f}".format(
                    epoch + 1, cfg.epochs, train_loss, val_loss
                )
            )
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_outcome_evaluation_scores["acc"])
            history["val_auroc"].append(val_outcome_evaluation_scores["auroc"])
            history["val_auprc"].append(val_outcome_evaluation_scores["auprc"])
            history["val_mad"].append(val_los_evaluation_scores["mad"])
            history["val_mse"].append(val_los_evaluation_scores["mse"])
            history["val_mape"].append(val_los_evaluation_scores["mape"])
            history["val_rmse"].append(val_los_evaluation_scores["rmse"])
            history["val_early_prediction_score"].append(
                val_covid_evaluation_scores["early_prediction_score"]
            )
            history["val_multitask_los_score"].append(
                val_covid_evaluation_scores["multitask_los_score"]
            )

            # if mad is lower, than set the best mad, save the model, and test it on the test set
            if val_los_evaluation_scores["mad"] < best_val_performance:
                best_val_performance = val_los_evaluation_scores["mad"]
                torch.save(model.state_dict(), f"checkpoints/{cfg.name}.pth")
        all_history["test_fold_{}".format(fold_test + 1)] = history
        print(
            f"Best performance on val set {fold_test+1}: \
            MAE = {best_val_performance}"
        )
        model = build_model_from_cfg(cfg)
        model.load_state_dict(torch.load(f"checkpoints/{cfg.name}.pth"))
        (
            test_loss,
            test_outcome_evaluation_scores,
            test_los_evaluation_scores,
            test_covid_evaluation_scores,
        ) = val_epoch(model, device, test_loader, criterion, los_statistics)
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
        "EarlyPredictionScore: {:.3f} ({:.3f})".format(
            test_early_prediction_list.mean(), test_early_prediction_list.std()
        )
    )
    print(
        "MultitaskPredictionScore: {:.3f} ({:.3f})".format(
            test_multitask_los_list.mean(), test_multitask_los_list.std()
        )
    )


def start_inference(cfg, device):
    dataset_type, method, num_folds, train_fold = (
        cfg.dataset,
        cfg.model,
        cfg.num_folds,
        cfg.train_fold,
    )
    # Load data
    x, y, x_lab_length = load_data(dataset_type)
    x, y, x_lab_length = x.float(), y.float(), x_lab_length.float()
    print(x.shape)

    model = build_model_from_cfg(cfg)
    model.load_state_dict(torch.load(f"checkpoints/{cfg.name}.pth"))

    idx = 4
    _, out = model(x[idx : idx + 1])
    los_statistics = {"los_mean": 7.3101974, "los_std": 6.9111395}
    out = out * los_statistics["los_std"] + los_statistics["los_mean"]
    # y = y * los_statistics["los_std"] + los_statistics["los_mean"]
    print("x_lab_length:", x_lab_length[idx : idx + 1])
    print("---- start y_true --------")
    print(y[idx : idx + 1])
    print("---- end y_true --------")
    print("---- start y_pred --------")
    print(out)
    print("---- end y_pred --------")
