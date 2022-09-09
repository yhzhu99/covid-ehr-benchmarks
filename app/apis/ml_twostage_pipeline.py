import copy
import math
import pathlib
import pickle
import random

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
    train_test_split,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from app.core.evaluation import covid_metrics, eval_metrics
from app.core.utils import init_random
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
    x, y_raw, len_list, outcome_model, los_model, cfg, los_statistics
):
    y = copy.deepcopy(y_raw)
    y_outcome_pred = outcome_model.predict(x)
    y_outcome_true = y[:, 0]
    y_los_true = y[:, 1]
    outcome_evaluation_scores = eval_metrics.print_metrics_binary(
        y_outcome_true, y_outcome_pred, verbose=0
    )

    y_los_pred = los_model.predict(x)
    y = reverse_zscore_los(y, los_statistics)
    y_los_pred = reverse_zscore_los(y_los_pred, los_statistics)

    los_evaluation_scores = eval_metrics.print_metrics_regression(
        y_los_true, y_los_pred, verbose=0
    )

    early_prediction_score = covid_metrics.early_prediction_outcome_metric(
        y, y_outcome_pred, len_list, cfg.thresholds, verbose=0
    )

    multitask_los_score = covid_metrics.multitask_los_metric(
        y,
        y_outcome_pred,
        y_los_pred,
        cfg.large_los,
        cfg.thresholds,
        verbose=0,
    )

    covid_evaluation_scores = {
        "early_prediction_score": early_prediction_score,
        "multitask_los_score": multitask_los_score,
    }
    return (
        outcome_evaluation_scores,
        los_evaluation_scores,
        covid_evaluation_scores,
    )


def calculate_los_statistics(y):
    """calculate los's mean/std"""
    mean, std = y.mean(), y.std()
    los_statistics = {"los_mean": mean, "los_std": std}
    return los_statistics


def zscore_los(y, los_statistics):
    """zscore scale y"""
    y[:, 1] = (y[:, 1] - los_statistics["los_mean"]) / los_statistics["los_std"]
    return y


def reverse_zscore_los(y, los_statistics):
    """reverse zscore y"""
    if len(y.shape) == 1:
        y = y * los_statistics["los_std"] + los_statistics["los_mean"]
    elif len(y.shape) == 2:  # [outcome, los]
        y[:, 1] = y[:, 1] * los_statistics["los_std"] + los_statistics["los_mean"]
    return y


def start_pipeline(cfg):
    dataset_type, method, num_folds, train_fold = (
        cfg.dataset,
        cfg.model,
        cfg.num_folds,
        cfg.train_fold,
    )
    # Load data
    x, y, x_lab_length = load_data(dataset_type)
    x, y_outcome, y_los, x_lab_length = numpy_dataset(x, y, x_lab_length)

    all_history = {}
    test_performance = {
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
        n_splits=num_folds, shuffle=True, random_state=cfg.dataset_split_seed
    )
    skf = kfold_test.split(np.arange(len(x)), y_outcome)
    for fold_test in range(train_fold):
        train_and_val_idx, test_idx = next(skf)
        print("====== Test Fold {} ======".format(fold_test + 1))
        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=1 / (num_folds - 1),
            random_state=cfg.dataset_split_seed,
        )
        sub_x = x[train_and_val_idx]
        sub_x_lab_length = x_lab_length[train_and_val_idx]
        sub_y = y[train_and_val_idx]
        sub_y_los = sub_y[:, :, 1]
        sub_y_outcome = sub_y[:, 0, 0]

        train_idx, val_idx = next(
            sss.split(np.arange(len(train_and_val_idx)), sub_y_outcome)
        )

        x_train, y_train, _ = flatten_dataset(
            sub_x, sub_y, train_idx, sub_x_lab_length, case="outcome"
        )

        los_statistics = calculate_los_statistics(y_train)
        print(los_statistics)

        (x_test, y_test, len_list_test) = flatten_dataset(
            x, y, test_idx, x_lab_length, case="outcome"
        )
        y_test = zscore_los(y_test, los_statistics)

        all_history["test_fold_{}".format(fold_test + 1)] = {}
        for seed in cfg.model_init_seed:
            outcome_model = pd.read_pickle(
                f"checkpoints/{cfg.outcome_model_name}_{fold_test + 1}_seed{seed}.pth"
            )
            los_model = pd.read_pickle(
                f"checkpoints/{cfg.los_model_name}_{fold_test + 1}_seed{seed}.pth"
            )
            (
                test_outcome_evaluation_scores,
                test_los_evaluation_scores,
                test_covid_evaluation_scores,
            ) = twostage_inference(
                x_test,
                y_test,
                len_list_test,
                outcome_model,
                los_model,
                cfg,
                los_statistics,
            )

            test_performance["test_mad"].append(test_los_evaluation_scores["mad"])
            test_performance["test_mse"].append(test_los_evaluation_scores["mse"])
            test_performance["test_mape"].append(test_los_evaluation_scores["mape"])
            test_performance["test_rmse"].append(test_los_evaluation_scores["rmse"])
            test_performance["test_accuracy"].append(
                test_outcome_evaluation_scores["acc"]
            )
            test_performance["test_auroc"].append(
                test_outcome_evaluation_scores["auroc"]
            )
            test_performance["test_auprc"].append(
                test_outcome_evaluation_scores["auprc"]
            )
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
