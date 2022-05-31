import math
import pathlib
import pickle
import random

import numpy as np
import pandas as pd
import xgboost as xgb
from autogluon.tabular import TabularPredictor
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
    train_test_split,
)
from sklearn.tree import DecisionTreeRegressor

from app import dataset
from app.dataset.ml import flatten_dataset, numpy_dataset
from app.utils import RANDOM_SEED, metrics


def train(x, y, method):
    if method == "xgboost":
        model = xgb.XGBRegressor(verbosity=0, n_estimators=1000, learning_rate=0.1)
        model.fit(x, y, eval_metric="auc")
    elif method == "gbdt":
        method = GradientBoostingRegressor(random_state=RANDOM_SEED)
        model = method.fit(x, y)
    elif method == "random_forest":
        method = RandomForestRegressor(random_state=RANDOM_SEED, max_depth=2)
        model = method.fit(x, y)
    elif method == "decision_tree":
        model = DecisionTreeRegressor(random_state=RANDOM_SEED)
        model.fit(x, y)
    elif method == "catboost":
        model = CatBoostRegressor(
            iterations=10,  # performance is better when iterations = 100
            learning_rate=0.1,
            depth=3,
            loss_function="RMSE",
            verbose=None,
            silent=True,
            allow_writing_files=False,
        )
        model.fit(x, y)
    return model


def validate(x, y, model):
    y_pred = model.predict(x)
    evaluation_scores = metrics.print_metrics_regression(y, y_pred, verbose=0)
    return evaluation_scores


def test(x, y, model):
    y_pred = model.predict(x)
    # print(y_pred[0:10], y[0:10])
    evaluation_scores = metrics.print_metrics_regression(y, y_pred, verbose=0)
    return evaluation_scores


def start_pipeline(cfg):
    dataset_type, mode, method, num_folds, train_fold = (
        cfg.dataset,
        cfg.mode,
        cfg.model,
        cfg.num_folds,
        cfg.train_fold,
    )
    # Load data
    x, y, x_lab_length = dataset.load_data(dataset_type)
    x, y_outcome, y_los, x_lab_length = numpy_dataset(x, y, x_lab_length)

    all_history = {}
    test_performance = {"test_mad": [], "test_mse": [], "test_mape": []}

    kfold_test = StratifiedKFold(
        n_splits=num_folds, shuffle=True, random_state=RANDOM_SEED
    )

    for fold_test in range(train_fold):
        train_and_val_idx, test_idx = next(
            kfold_test.split(np.arange(len(x)), y_outcome)
        )
        print("====== Test Fold {} ======".format(fold_test + 1))
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=1 / (num_folds - 1), random_state=RANDOM_SEED
        )

        sub_x = x[train_and_val_idx]
        sub_x_lab_length = x_lab_length[train_and_val_idx]
        sub_y = y[train_and_val_idx]
        sub_y_los = sub_y[:, :, 1]
        sub_y_outcome = sub_y[:, 0, 0]

        train_idx, val_idx = next(
            sss.split(np.arange(len(train_and_val_idx)), sub_y_outcome)
        )

        x_train, y_train = flatten_dataset(sub_x, sub_y, train_idx, sub_x_lab_length)
        x_val, y_val = flatten_dataset(sub_x, sub_y, val_idx, sub_x_lab_length)
        x_test, y_test = flatten_dataset(x, y, test_idx, x_lab_length)

        all_history["test_fold_{}".format(fold_test + 1)] = {}

        model = train(x_train, y_train, method)

        if mode == "val":
            history = {"val_mad": [], "val_mse": [], "val_mape": []}
            val_evaluation_scores = validate(x_val, y_val, model)
            history["val_mad"].append(val_evaluation_scores["mad"])
            history["val_mse"].append(val_evaluation_scores["mse"])
            history["val_mape"].append(val_evaluation_scores["mape"])
            all_history["test_fold_{}".format(fold_test + 1)] = history
            print(
                f"Performance on val set {fold_test+1}: \
                MAE = {val_evaluation_scores['mad']}, \
                MSE = {val_evaluation_scores['mse']}, \
                MAPE = {val_evaluation_scores['mape']}"
            )

        elif mode == "test":
            test_evaluation_scores = test(x_test, y_test, model)
            test_performance["test_mad"].append(test_evaluation_scores["mad"])
            test_performance["test_mse"].append(test_evaluation_scores["mse"])
            test_performance["test_mape"].append(test_evaluation_scores["mape"])
            print(
                f"Performance on test set {fold_test+1}: \
                MAE = {test_evaluation_scores['mad']}, \
                MSE = {test_evaluation_scores['mse']}, \
                MAPE = {test_evaluation_scores['mape']}"
            )

            # Calculate average performance on 10-fold test set
            test_mad_list = np.array(test_performance["test_mad"])
            test_mse_list = np.array(test_performance["test_mse"])
            test_mape_list = np.array(test_performance["test_mape"])
    if mode == "test":
        print("====================== TEST RESULT ======================")
        print("MAE: {:.3f} ({:.3f})".format(test_mad_list.mean(), test_mad_list.std()))
        print("MSE: {:.3f} ({:.3f})".format(test_mse_list.mean(), test_mse_list.std()))
        print(
            "MAPE: {:.3f} ({:.3f})".format(test_mape_list.mean(), test_mape_list.std())
        )
