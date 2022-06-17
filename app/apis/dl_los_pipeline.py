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

from app.core.evaluation import eval_metrics
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
        batch_y = batch_y[:, :, 1]  # 0: outcome, 1: los
        batch_y = batch_y.unsqueeze(-1)
        optimizer.zero_grad()
        output = model(batch_x)
        loss = loss_fn(output, batch_y, batch_x_lab_length)
        train_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    return np.array(train_loss).mean()


def val_epoch(model, device, dataloader, loss_fn, los_statistics):
    """
    val / test
    """
    val_loss = []
    y_pred = []
    y_true = []
    model.eval()
    with torch.no_grad():
        for step, data in enumerate(dataloader):
            batch_x, batch_y, batch_x_lab_length = data
            batch_x, batch_y, batch_x_lab_length = (
                batch_x.float(),
                batch_y.float(),
                batch_x_lab_length.float(),
            )
            batch_y = batch_y[:, :, 1]  # 0: outcome, 1: los
            batch_y = batch_y.unsqueeze(-1)
            output = model(batch_x)
            loss = loss_fn(output, batch_y, batch_x_lab_length)
            val_loss.append(loss.item())
            output = torch.squeeze(output)
            batch_y = torch.squeeze(batch_y)
            for i in range(len(batch_y)):
                y_pred.extend(output[i][: batch_x_lab_length[i].long()].tolist())
                y_true.extend(batch_y[i][: batch_x_lab_length[i].long()].tolist())
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    y_pred = reverse_zscore_los(y_pred, los_statistics)
    y_true = reverse_zscore_los(y_true, los_statistics)
    evaluation_scores = eval_metrics.print_metrics_regression(y_true, y_pred, verbose=1)
    return np.array(val_loss).mean(), evaluation_scores


def calculate_los_statistics(dataset, train_idx):
    """calculate los's mean/std"""
    # y = dataset.y[train_idx][:, :, 1]
    y = []
    for i in train_idx:
        # print(dataset.y[i][:dataset.x_lab_length[i].long()])
        for j in range(dataset.x_lab_length[i]):
            y.append(dataset.y[i][j][1])
        # y.extend(dataset.y[i][:dataset.x_lab_length[i].long()].tolist())
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
    }
    kfold_test = StratifiedKFold(
        n_splits=num_folds, shuffle=True, random_state=RANDOM_SEED
    )
    skf = kfold_test.split(np.arange(len(dataset)), dataset.y[:, 0, 0])
    for fold_test in range(train_fold):
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
        criterion = predict_all_visits_mse_loss
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_mad": [],
            "val_mse": [],
            "val_mape": [],
        }
        best_val_performance = 1e8
        for epoch in range(cfg.epochs):
            train_loss = train_epoch(model, device, train_loader, criterion, optimizer)
            val_loss, val_evaluation_scores = val_epoch(
                model, device, val_loader, criterion, los_statistics
            )
            # save performance history on validation set
            print(
                "Epoch:{}/{} AVG Training Loss:{:.3f} AVG Val Loss:{:.3f}".format(
                    epoch + 1, cfg.epochs, train_loss, val_loss
                )
            )
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_mad"].append(val_evaluation_scores["mad"])
            history["val_mse"].append(val_evaluation_scores["mse"])
            history["val_mape"].append(val_evaluation_scores["mape"])
            # if mad is lower, than set the best mad, save the model, and test it on the test set
            if val_evaluation_scores["mad"] < best_val_performance:
                best_val_performance = val_evaluation_scores["mad"]
                torch.save(model.state_dict(), f"checkpoints/{cfg.name}.pth")
        all_history["test_fold_{}".format(fold_test + 1)] = history
        print(
            f"Best performance on val set {fold_test+1}: \
            MAE = {best_val_performance}"
        )
        model = build_model_from_cfg(cfg)
        model.load_state_dict(torch.load(f"checkpoints/{cfg.name}.pth"))
        test_loss, test_evaluation_scores = val_epoch(
            model, device, test_loader, criterion, los_statistics
        )
        test_performance["test_loss"].append(test_loss)
        test_performance["test_mad"].append(test_evaluation_scores["mad"])
        test_performance["test_mse"].append(test_evaluation_scores["mse"])
        test_performance["test_mape"].append(test_evaluation_scores["mape"])
        print(
            f"Performance on test set {fold_test+1}: MAE = {test_evaluation_scores['mad']}, MSE = {test_evaluation_scores['mse']}, MAPE = {test_evaluation_scores['mape']}"
        )
    # Calculate average performance on 10-fold test set
    test_mad_list = np.array(test_performance["test_mad"])
    test_mse_list = np.array(test_performance["test_mse"])
    test_mape_list = np.array(test_performance["test_mape"])
    print("MAE: {:.3f} ({:.3f})".format(test_mad_list.mean(), test_mad_list.std()))
    print("MSE: {:.3f} ({:.3f})".format(test_mse_list.mean(), test_mse_list.std()))
    print("MAPE: {:.3f} ({:.3f})".format(test_mape_list.mean(), test_mape_list.std()))
