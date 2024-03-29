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


def train(x, y, method, cfg, seed=42):
    y = y[:, 0]

    if method == "xgboost":
        model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            verbosity=0,
            learning_rate=cfg.learning_rate,
            max_depth=cfg.max_depth,
            min_child_weight=cfg.min_child_weight,
            n_estimators=1000,
            use_label_encoder=False,
            random_state=seed,
        )
        model.fit(x, y)
    elif method == "gbdt":
        method = GradientBoostingClassifier(
            random_state=seed,
            learning_rate=cfg.learning_rate,
            n_estimators=cfg.n_estimators,
            subsample=cfg.subsample,
        )
        model = method.fit(x, y)
    elif method == "random_forest":
        method = RandomForestClassifier(
            random_state=seed,
            max_depth=cfg.max_depth,
            min_samples_split=cfg.min_samples_split,
            n_estimators=cfg.n_estimators,
        )
        model = method.fit(x, y)
    elif method == "decision_tree":
        model = DecisionTreeClassifier(random_state=seed, max_depth=cfg.max_depth)
        model.fit(x, y)
    elif method == "catboost":
        model = CatBoostClassifier(
            random_seed=seed,
            iterations=cfg.iterations,  # performance is better when iterations = 100
            learning_rate=cfg.learning_rate,
            depth=cfg.depth,
            verbose=None,
            silent=True,
            allow_writing_files=False,
            loss_function="CrossEntropy",
        )
        model.fit(x, y)
    return model


def validate(x, y, len_list, model, cfg):
    """val/test"""
    y_outcome_pred = model.predict(x)
    y_outcome_true = y[:, 0]
    evaluation_scores = eval_metrics.print_metrics_binary(
        y_outcome_true, y_outcome_pred, verbose=0
    )
    early_prediction_score = covid_metrics.early_prediction_outcome_metric(
        y, y_outcome_pred, len_list, cfg.thresholds, verbose=0
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

        x_train, y_train, len_list_train = flatten_dataset(
            sub_x, sub_y, train_idx, sub_x_lab_length, case="outcome"
        )
        x_val, y_val, len_list_val = flatten_dataset(
            sub_x, sub_y, val_idx, sub_x_lab_length, case="outcome"
        )
        x_test, y_test, len_list_test = flatten_dataset(
            x, y, test_idx, x_lab_length, case="outcome"
        )
        all_history["test_fold_{}".format(fold_test + 1)] = {}
        history = {
            "val_accuracy": [],
            "val_auroc": [],
            "val_auprc": [],
            "val_early_prediction_score": [],
        }
        for seed in cfg.model_init_seed:
            init_random(seed)
            if cfg.train == True:
                model = train(x_train, y_train, method, cfg, seed)
                pd.to_pickle(
                    model, f"checkpoints/{cfg.name}_{fold_test + 1}_seed{seed}.pth"
                )
            if mode == "val":
                val_evaluation_scores = validate(x_val, y_val, len_list_val, model, cfg)
                history["val_accuracy"].append(val_evaluation_scores["acc"])
                history["val_auroc"].append(val_evaluation_scores["auroc"])
                history["val_auprc"].append(val_evaluation_scores["auprc"])
                history["val_early_prediction_score"].append(
                    val_evaluation_scores["early_prediction_score"]
                )
                print(
                    f"Performance on val set {fold_test+1}: \
                    ACC = {val_evaluation_scores['acc']}, \
                    AUROC = {val_evaluation_scores['auroc']}, \
                    AUPRC = {val_evaluation_scores['auprc']}, \
                    EarlyPredictionScore = {val_evaluation_scores['early_prediction_score']}"
                )
            elif mode == "test":
                model = pd.read_pickle(
                    f"checkpoints/{cfg.name}_{fold_test + 1}_seed{seed}.pth"
                )
                test_evaluation_scores = validate(
                    x_test, y_test, len_list_test, model, cfg
                )
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
        all_history["test_fold_{}".format(fold_test + 1)] = history
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
        print("=========================================================")
        perflog.process_and_upload_performance(
            cfg,
            acc=val_accuracy_list,
            auroc=val_auroc_list,
            auprc=val_auprc_list,
            early_prediction_score=val_early_prediction_list,
            verbose=1,
            upload=cfg.db,
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

        print("=========================================================")
        perflog.process_and_upload_performance(
            cfg,
            acc=test_accuracy_list,
            auroc=test_auroc_list,
            auprc=test_auprc_list,
            early_prediction_score=test_early_prediction_list,
            verbose=1,
            upload=cfg.db,
        )
