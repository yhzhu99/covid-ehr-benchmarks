import math
import pathlib
import pickle
import random

import numpy as np
import pandas as pd
import xgboost as xgb
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

from app.core.evaluation import eval_metrics
from app.core.utils import init_random
from app.datasets.base import load_data
from app.datasets.dl import Dataset
from app.datasets.ml import flatten_dataset, numpy_dataset
from app.models import (
    build_model_from_cfg,
    get_multi_task_loss,
    predict_all_visits_bce_loss,
    predict_all_visits_mse_loss,
)
from app.utils import perflog


def train(x, y, method, cfg, seed=42):
    if method == "xgboost":
        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            eval_metric="error",
            verbosity=0,
            learning_rate=cfg.learning_rate,
            max_depth=cfg.max_depth,
            min_child_weight=cfg.min_child_weight,
            n_estimators=1000,
            use_label_encoder=False,
            random_state=seed,
        )
        model.fit(x, y, eval_metric="mae")
    elif method == "gbdt":
        method = GradientBoostingRegressor(
            random_state=seed,
            learning_rate=cfg.learning_rate,
            n_estimators=cfg.n_estimators,
            subsample=cfg.subsample,
        )
        model = method.fit(x, y)
    elif method == "random_forest":
        method = RandomForestRegressor(
            random_state=seed,
            max_depth=cfg.max_depth,
            min_samples_split=cfg.min_samples_split,
            n_estimators=cfg.n_estimators,
        )
        model = method.fit(x, y)
    elif method == "decision_tree":
        model = DecisionTreeRegressor(random_state=seed, max_depth=cfg.max_depth)
        model.fit(x, y)
    elif method == "catboost":
        model = CatBoostRegressor(
            random_seed=seed,
            iterations=cfg.iterations,  # performance is better when iterations = 100
            learning_rate=cfg.learning_rate,
            depth=cfg.depth,
            verbose=None,
            silent=True,
            allow_writing_files=False,
            loss_function="MAE",
        )
        model.fit(x, y)
    return model


def validate(x, y, model, los_statistics):
    """val/test"""
    y_pred = model.predict(x)
    y = reverse_zscore_los(y, los_statistics)
    y_pred = reverse_zscore_los(y_pred, los_statistics)
    evaluation_scores = eval_metrics.print_metrics_regression(y, y_pred, verbose=0)
    return evaluation_scores


def calculate_los_statistics(y):
    """calculate los's mean/std"""
    mean, std = y.mean(), y.std()
    los_statistics = {"los_mean": mean, "los_std": std}
    return los_statistics


def zscore_los(y, los_statistics):
    """zscore scale y"""
    y = (y - los_statistics["los_mean"]) / los_statistics["los_std"]
    return y


def reverse_zscore_los(y, los_statistics):
    """reverse zscore y"""
    y = y * los_statistics["los_std"] + los_statistics["los_mean"]
    return y


def start_pipeline(cfg):
    dataset_type, mode, method, num_folds, train_fold = (
        cfg.dataset,
        cfg.mode,
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

        x_train, y_train = flatten_dataset(
            sub_x, sub_y, train_idx, sub_x_lab_length, case="los"
        )

        los_statistics = calculate_los_statistics(y_train)
        print(los_statistics)
        y_train = zscore_los(y_train, los_statistics)

        x_val, y_val = flatten_dataset(
            sub_x, sub_y, val_idx, sub_x_lab_length, case="los"
        )
        y_val = zscore_los(y_val, los_statistics)

        x_test, y_test = flatten_dataset(x, y, test_idx, x_lab_length, case="los")
        y_test = zscore_los(y_test, los_statistics)

        all_history["test_fold_{}".format(fold_test + 1)] = {}
        history = {"val_mad": [], "val_mse": [], "val_mape": [], "val_rmse": []}
        for seed in cfg.model_init_seed:
            init_random(seed)
            model = train(x_train, y_train, method, cfg, seed)
            if mode == "val":
                val_evaluation_scores = validate(x_val, y_val, model, los_statistics)
                history["val_mad"].append(val_evaluation_scores["mad"])
                history["val_mse"].append(val_evaluation_scores["mse"])
                history["val_mape"].append(val_evaluation_scores["mape"])
                history["val_rmse"].append(val_evaluation_scores["rmse"])
                print(
                    f"Performance on val set {fold_test+1}: \
                    MAE = {val_evaluation_scores['mad']}, \
                    MSE = {val_evaluation_scores['mse']}, \
                    MAPE = {val_evaluation_scores['mape']},\
                    RMSE = {val_evaluation_scores['rmse']}"
                )
            elif mode == "test":
                test_evaluation_scores = validate(x_test, y_test, model, los_statistics)
                test_performance["test_mad"].append(test_evaluation_scores["mad"])
                test_performance["test_mse"].append(test_evaluation_scores["mse"])
                test_performance["test_mape"].append(test_evaluation_scores["mape"])
                test_performance["test_rmse"].append(test_evaluation_scores["rmse"])
                print(
                    f"Performance on test set {fold_test+1}: \
                    MAE = {test_evaluation_scores['mad']}, \
                    MSE = {test_evaluation_scores['mse']}, \
                    MAPE = {test_evaluation_scores['mape']}, \
                    RMSE = {test_evaluation_scores['rmse']}"
                )
            all_history["test_fold_{}".format(fold_test + 1)] = history
    if mode == "val":
        # Calculate average performance on 10-fold val set
        val_mad_list = []
        val_mse_list = []
        val_mape_list = []
        val_rmse_list = []
        for f in range(train_fold):
            val_mad_list.extend(all_history[f"test_fold_{f + 1}"]["val_mad"])
            val_mse_list.extend(all_history[f"test_fold_{f + 1}"]["val_mse"])
            val_mape_list.extend(all_history[f"test_fold_{f + 1}"]["val_mape"])
            val_rmse_list.extend(all_history[f"test_fold_{f + 1}"]["val_rmse"])
        val_mad_list = np.array(val_mad_list)
        val_mse_list = np.array(val_mse_list)
        val_mape_list = np.array(val_mape_list)
        val_rmse_list = np.array(val_rmse_list)
        print("====================== VAL RESULT ======================")
        print("MAE: {:.3f} ({:.3f})".format(val_mad_list.mean(), val_mad_list.std()))
        print("MSE: {:.3f} ({:.3f})".format(val_mse_list.mean(), val_mse_list.std()))
        print("MAPE: {:.3f} ({:.3f})".format(val_mape_list.mean(), val_mape_list.std()))
        print("RMSE: {:.3f} ({:.3f})".format(val_rmse_list.mean(), val_rmse_list.std()))
    elif mode == "test":
        # Calculate average performance on 10-fold test set
        test_mad_list = np.array(test_performance["test_mad"])
        test_mse_list = np.array(test_performance["test_mse"])
        test_mape_list = np.array(test_performance["test_mape"])
        test_rmse_list = np.array(test_performance["test_rmse"])
        print("====================== TEST RESULT ======================")
        print("MAE: {:.3f} ({:.3f})".format(test_mad_list.mean(), test_mad_list.std()))
        print("MSE: {:.3f} ({:.3f})".format(test_mse_list.mean(), test_mse_list.std()))
        print(
            "MAPE: {:.3f} ({:.3f})".format(test_mape_list.mean(), test_mape_list.std())
        )
        print(
            "RMSE: {:.3f} ({:.3f})".format(test_rmse_list.mean(), test_rmse_list.std())
        )

        print("=========================================================")
        perflog.process_and_upload_performance(
            cfg,
            mae=test_mad_list,
            mse=test_mse_list,
            rmse=test_rmse_list,
            mape=test_mape_list,
            verbose=1,
            upload=cfg.db,
        )
