import math
import pathlib
import pickle
import random

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from autogluon.tabular import TabularPredictor
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

from app.utils import metrics

"""
Tasks:
- mortality outcome
- los

Models:
- random forest (sklearn)
- xgboost (xgboost)
- catboost (catboost)
- gbdt (sklearn)
- autogluon (automl models)
"""

RANDOM_SEED = 42


def flatten_dataset(x, y, indices, visits_length):
    x_flat = []
    y_flat = []
    for i in indices:
        for v in range(visits_length[i]):
            x_flat.append(x[i][v])
            y_flat.append(y[i][v][0])
    return np.array(x_flat), np.array(y_flat)


def train(x, y, method):
    if method == "xgboost":
        model = xgb.XGBClassifier(
            verbosity=0, n_estimators=1000, learning_rate=0.1, use_label_encoder=False
        )
        model.fit(x, y, eval_metric="auc")
    elif method == "gbdt":
        method = GradientBoostingClassifier(
            n_estimators=100, learning_rate=1.0, max_depth=1, random_state=RANDOM_SEED
        )
        model = method.fit(x, y)
    elif method == "random_forest":
        method = RandomForestClassifier(random_state=RANDOM_SEED, max_depth=2)
        model = method.fit(x, y)
    elif method == "decision_tree":
        model = DecisionTreeClassifier(random_state=RANDOM_SEED)
        model.fit(x, y)
    elif method == "catboost":
        model = CatBoostClassifier(
            iterations=10,  # performance is better when iterations = 100
            learning_rate=0.1,
            depth=3,
            verbose=None,
            silent=True,
            allow_writing_files=False,
        )
        model.fit(x, y)
    return model


def validate(x, y, model):
    y_pred = model.predict(x)
    evaluation_scores = metrics.print_metrics_binary(y, y_pred, verbose=0)
    return evaluation_scores


def test(x, y, model):
    y_pred = model.predict(x)
    # print(y_pred[0:10], y[0:10])
    evaluation_scores = metrics.print_metrics_binary(y, y_pred, verbose=0)
    return evaluation_scores


if __name__ == "__main__":
    data_path = "./dataset/hm/processed_data/"
    x = pickle.load(open(data_path + "x.pkl", "rb"))
    y = pickle.load(open(data_path + "y.pkl", "rb"))
    x_lab_length = pickle.load(open(data_path + "visits_length.pkl", "rb"))

    x = x.numpy()
    y = y.numpy()
    x_lab_length = x_lab_length.numpy()
    y_los = y[:, :, 1]
    y_outcome = y[:, 0, 0]

    num_folds = 10
    method = "catboost"
    mode = "test"  # val / test

    all_history = {}
    test_performance = {"test_accuracy": [], "test_auroc": [], "test_auprc": []}

    kfold_test = StratifiedKFold(
        n_splits=num_folds, shuffle=True, random_state=RANDOM_SEED
    )
    for fold_test, (train_and_val_idx, test_idx) in enumerate(
        kfold_test.split(np.arange(len(x)), y_outcome)
    ):
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
            history = {
                "val_accuracy": [],
                "val_auroc": [],
                "val_auprc": [],
            }
            val_evaluation_scores = validate(x_val, y_val, model)
            history["val_accuracy"].append(val_evaluation_scores["acc"])
            history["val_auroc"].append(val_evaluation_scores["auroc"])
            history["val_auprc"].append(val_evaluation_scores["auprc"])
            all_history["test_fold_{}".format(fold_test + 1)] = history
            print(
                f"Performance on val set {fold_test+1}: \
                ACC = {val_evaluation_scores['acc']}, \
                AUROC = {val_evaluation_scores['auroc']}, \
                AUPRC = {val_evaluation_scores['auprc']}"
            )

        elif mode == "test":
            test_evaluation_scores = test(x_test, y_test, model)
            test_performance["test_accuracy"].append(test_evaluation_scores["acc"])
            test_performance["test_auroc"].append(test_evaluation_scores["auroc"])
            test_performance["test_auprc"].append(test_evaluation_scores["auprc"])
            print(
                f"Performance on test set {fold_test+1}: \
                ACC = {test_evaluation_scores['acc']}, \
                AUROC = {test_evaluation_scores['auroc']}, \
                AUPRC = {test_evaluation_scores['auprc']}"
            )

            # Calculate average performance on 10-fold test set
            test_accuracy_list = np.array(test_performance["test_accuracy"])
            test_auroc_list = np.array(test_performance["test_auroc"])
            test_auprc_list = np.array(test_performance["test_auprc"])
    if mode == "test":
        print("====================== TEST RESULT ======================")
        print(
            "ACC: {:.3f} ({:.3f})".format(
                test_accuracy_list.mean(), test_accuracy_list.std()
            )
        )
        print(
            "AUROC: {:.3f} ({:.3f})".format(
                test_auroc_list.mean(), test_auroc_list.std()
            )
        )
        print(
            "AUPRC: {:.3f} ({:.3f})".format(
                test_auprc_list.mean(), test_auprc_list.std()
            )
        )
