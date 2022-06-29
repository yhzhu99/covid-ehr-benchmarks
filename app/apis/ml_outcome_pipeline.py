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

from app.core.evaluation import covid_metrics, eval_metrics
from app.core.utils import RANDOM_SEED, init_random
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


def train(x, y, method):
    y = y[:, 0]

    if method == "xgboost":
        model = xgb.XGBClassifier(
            verbosity=0, n_estimators=1000, learning_rate=0.3, use_label_encoder=False
        )
        model.fit(x, y, eval_metric="auc")
    elif method == "gbdt":
        method = GradientBoostingClassifier(
            n_estimators=100, learning_rate=1.0, max_depth=1, random_state=RANDOM_SEED
        )
        model = method.fit(x, y)
    elif method == "random_forest":
        method = RandomForestClassifier(random_state=RANDOM_SEED, max_depth=100)
        model = method.fit(x, y)
    elif method == "decision_tree":
        model = DecisionTreeClassifier(random_state=RANDOM_SEED)
        model.fit(x, y)
    elif method == "catboost":
        model = CatBoostClassifier(
            iterations=50,  # performance is better when iterations = 100
            learning_rate=0.5,
            depth=3,
            verbose=None,
            silent=True,
            allow_writing_files=False,
        )
        model.fit(x, y)
    return model


def validate(x, y, model, cfg):
    """val/test"""
    y_outcome_pred = model.predict(x)
    y_outcome_true = y[:, 0]
    evaluation_scores = eval_metrics.print_metrics_binary(
        y_outcome_true, y_outcome_pred, verbose=0
    )
    early_prediction_score = covid_metrics.early_prediction_outcome_metric(
        y, y_outcome_pred, cfg.thresholds, verbose=0
    )
    evaluation_scores["early_prediction_score"] = early_prediction_score
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
    x, y, x_lab_length = load_data(dataset_type)
    x, y_outcome, y_los, x_lab_length = numpy_dataset(x, y, x_lab_length)

    all_history = {}
    test_performance = {
        "test_accuracy": [],
        "test_auroc": [],
        "test_auprc": [],
        "test_early_prediction_score": [],
    }

    kfold_test = StratifiedKFold(
        n_splits=num_folds, shuffle=True, random_state=RANDOM_SEED
    )
    skf = kfold_test.split(np.arange(len(x)), y_outcome)
    for fold_test in range(train_fold):
        train_and_val_idx, test_idx = next(skf)
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

        x_train, y_train = flatten_dataset(
            sub_x, sub_y, train_idx, sub_x_lab_length, case="outcome"
        )
        x_val, y_val = flatten_dataset(
            sub_x, sub_y, val_idx, sub_x_lab_length, case="outcome"
        )
        x_test, y_test = flatten_dataset(x, y, test_idx, x_lab_length, case="outcome")
        all_history["test_fold_{}".format(fold_test + 1)] = {}

        model = train(x_train, y_train, method)

        if mode == "val":
            history = {
                "val_accuracy": [],
                "val_auroc": [],
                "val_auprc": [],
                "val_early_prediction_score": [],
            }
            val_evaluation_scores = validate(x_val, y_val, model, cfg)

            history["val_accuracy"].append(val_evaluation_scores["acc"])
            history["val_auroc"].append(val_evaluation_scores["auroc"])
            history["val_auprc"].append(val_evaluation_scores["auprc"])
            history["val_early_prediction_score"].append(
                val_evaluation_scores["early_prediction_score"]
            )
            all_history["test_fold_{}".format(fold_test + 1)] = history
            print(
                f"Performance on val set {fold_test+1}: \
                ACC = {val_evaluation_scores['acc']}, \
                AUROC = {val_evaluation_scores['auroc']}, \
                AUPRC = {val_evaluation_scores['auprc']}, \
                EarlyPredictionScore = {val_evaluation_scores['early_prediction_score']}"
            )

        elif mode == "test":
            test_evaluation_scores = validate(x_test, y_test, model, cfg)
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
    if mode == "val":
        # Calculate average performance on 10-fold val set
        val_accuracy_list = []
        val_auroc_list = []
        val_auprc_list = []
        val_early_prediction_list = []
        for f in range(train_fold):
            val_accuracy_list.extend(all_history[f"test_fold_{f + 1}"]["val_accuracy"])
            val_auroc_list.extend(all_history[f"test_fold_{f + 1}"]["val_auroc"])
            val_auprc_list.extend(all_history[f"test_fold_{f + 1}"]["val_auprc"])
            val_early_prediction_list.extend(
                all_history[f"test_fold_{f + 1}"]["val_early_prediction_score"]
            )
        val_accuracy_list = np.array(val_accuracy_list)
        val_auroc_list = np.array(val_auroc_list)
        val_auprc_list = np.array(val_auprc_list)
        val_early_prediction_list = np.array(val_early_prediction_list)
        print("====================== VAL RESULT ======================")
        print(
            "ACC: {:.3f} ({:.3f})".format(
                val_accuracy_list.mean(), val_accuracy_list.std()
            )
        )
        print(
            "AUROC: {:.3f} ({:.3f})".format(val_auroc_list.mean(), val_auroc_list.std())
        )
        print(
            "AUPRC: {:.3f} ({:.3f})".format(val_auprc_list.mean(), val_auprc_list.std())
        )
        print(
            "EarlyPredictionScore:",
            (
                val_early_prediction_list.mean(axis=0),
                val_early_prediction_list.std(axis=0),
            ),
        )
    elif mode == "test":
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
            "AUROC: {:.3f} ({:.3f})".format(
                test_auroc_list.mean(), test_auroc_list.std()
            )
        )
        print(
            "AUPRC: {:.3f} ({:.3f})".format(
                test_auprc_list.mean(), test_auprc_list.std()
            )
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
        perflog.process_performance_raw_info(
            cfg,
            acc=test_accuracy_list,
            auroc=test_auroc_list,
            auprc=test_auprc_list,
            early_prediction_score=test_early_prediction_list,
            verbose=1,
        )
        perflog.process_and_upload_performance(
            cfg,
            acc=test_accuracy_list,
            auroc=test_auroc_list,
            auprc=test_auprc_list,
            early_prediction_score=test_early_prediction_list,
            verbose=0,
        )
