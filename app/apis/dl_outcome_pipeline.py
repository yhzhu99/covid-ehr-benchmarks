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
from app.utils import perflog


def train_epoch(model, device, dataloader, loss_fn, optimizer, info):
    train_loss = []
    model.train()
    for step, data in enumerate(dataloader):
        batch_x, batch_y, batch_x_lab_length = data
        batch_x, batch_y, batch_x_lab_length = (
            batch_x.float().to(device),
            batch_y.float().to(device),
            batch_x_lab_length.float().to(device),
        )
        batch_y = batch_y[:, :, 0]  # 0: outcome, 1: los
        batch_y = batch_y.unsqueeze(-1)
        optimizer.zero_grad()
        output = model(batch_x, info)
        loss = loss_fn(output, batch_y, batch_x_lab_length)
        train_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    return np.array(train_loss).mean()


def val_epoch(model, device, dataloader, loss_fn, info):
    """
    val / test
    """
    val_loss = []
    y_pred = []
    y_true = []
    y_true_all = []
    model.eval()
    with torch.no_grad():
        for step, data in enumerate(dataloader):
            batch_x, batch_y, batch_x_lab_length = data
            batch_x, batch_y, batch_x_lab_length = (
                batch_x.float().to(device),
                batch_y.float().to(device),
                batch_x_lab_length.float().to(device),
            )
            all_y = batch_y
            batch_y = batch_y[:, :, 0]  # 0: outcome, 1: los
            batch_y = batch_y.unsqueeze(-1)
            output = model(batch_x, info)
            loss = loss_fn(output, batch_y, batch_x_lab_length)
            val_loss.append(loss.item())
            for i in range(len(batch_y)):
                y_pred.extend(output[i][: batch_x_lab_length[i].long()].tolist())
                y_true.extend(batch_y[i][: batch_x_lab_length[i].long()].tolist())
                y_true_all.extend(all_y[i][: batch_x_lab_length[i].long()].tolist())
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_true_all = np.array(y_true_all)
    early_prediction_score = covid_metrics.early_prediction_outcome_metric(
        y_true_all, y_pred, info["config"].thresholds, verbose=0
    )
    y_pred = np.stack([1 - y_pred, y_pred], axis=1)
    evaluation_scores = eval_metrics.print_metrics_binary(y_true, y_pred, verbose=0)
    evaluation_scores["early_prediction_score"] = early_prediction_score
    return np.array(val_loss).mean(), evaluation_scores


def start_pipeline(cfg, device):
    info = {"config": cfg, "epoch": 0}
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
    model = build_model_from_cfg(cfg, device)
    print(model)
    all_history = {}
    test_performance = {
        "test_loss": [],
        "test_accuracy": [],
        "test_auroc": [],
        "test_auprc": [],
        "test_early_prediction_score": [],
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

        test_sampler = SubsetRandomSampler(test_idx)
        test_loader = DataLoader(
            dataset, batch_size=cfg.batch_size, sampler=test_sampler
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

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(
            dataset, batch_size=cfg.batch_size, sampler=train_sampler
        )
        val_loader = DataLoader(dataset, batch_size=cfg.batch_size, sampler=val_sampler)
        model = build_model_from_cfg(cfg, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = predict_all_visits_bce_loss
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_auroc": [],
            "val_auprc": [],
            "val_early_prediction_score": [],
        }
        best_val_performance = 0.0
        for epoch in range(cfg.epochs):
            info["epoch"] = epoch + 1
            train_loss = train_epoch(
                model,
                device,
                train_loader,
                criterion,
                optimizer,
                info=info,
            )
            val_loss, val_evaluation_scores = val_epoch(
                model,
                device,
                val_loader,
                criterion,
                info=val_info,
            )
            # save performance history on validation set
            print(
                "Epoch:{}/{} AVG Training Loss:{:.3f} AVG Val Loss:{:.3f}".format(
                    epoch + 1, cfg.epochs, train_loss, val_loss
                )
            )
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_evaluation_scores["acc"])
            history["val_auroc"].append(val_evaluation_scores["auroc"])
            history["val_auprc"].append(val_evaluation_scores["auprc"])
            history["val_early_prediction_score"].append(
                val_evaluation_scores["early_prediction_score"]
            )
            # if auroc is better, than set the best auroc, save the model, and test it on the test set
            if val_evaluation_scores["auprc"] > best_val_performance:
                best_val_performance = val_evaluation_scores["auprc"]
                torch.save(model.state_dict(), f"checkpoints/{cfg.name}.pth")
        all_history["test_fold_{}".format(fold_test + 1)] = history
        print(
            f"Best performance on val set {fold_test+1}: \
            AUPRC = {best_val_performance}"
        )
        model = build_model_from_cfg(cfg, device)
        model.load_state_dict(torch.load(f"checkpoints/{cfg.name}.pth"))
        test_loss, test_evaluation_scores = val_epoch(
            model,
            device,
            test_loader,
            criterion,
            info=val_info,
        )
        test_performance["test_loss"].append(test_loss)
        test_performance["test_accuracy"].append(test_evaluation_scores["acc"])
        test_performance["test_auroc"].append(test_evaluation_scores["auroc"])
        test_performance["test_auprc"].append(test_evaluation_scores["auprc"])
        test_performance["test_early_prediction_score"].append(
            test_evaluation_scores["early_prediction_score"]
        )
        print(
            f"Performance on test set {fold_test+1}: \
            ACC = {test_evaluation_scores['acc']}, \
            AUROC = {test_evaluation_scores['auroc']}, \
            AUPRC = {test_evaluation_scores['auprc']}, \
            EarlyPredictionScore = {test_evaluation_scores['early_prediction_score']}"
        )
    # Calculate average performance on 10-fold test set
    test_accuracy_list = np.array(test_performance["test_accuracy"])
    test_auroc_list = np.array(test_performance["test_auroc"])
    test_auprc_list = np.array(test_performance["test_auprc"])
    test_early_prediction_list = np.array(
        test_performance["test_early_prediction_score"]
    )

    print("====================== TEST RESULT ======================")
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
    print("=========================================================")
    perflog.process_and_upload_performance(
        cfg,
        acc=test_accuracy_list,
        auroc=test_auroc_list,
        auprc=test_auprc_list,
        early_prediction_score=test_early_prediction_list,
        verbose=1,
        upload=True,
    )
