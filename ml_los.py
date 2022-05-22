import math
import pathlib
import pickle

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from autogluon.tabular import TabularPredictor
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeRegressor

from app.utils import metrics

"""
Tasks:
- mortality outcome
- los

Models:
- logistic regression (sklearn) !TOFIX
- random forest (sklearn)
- xgboost (xgboost)
- catboost (catboost)
- gbdt (sklearn)
- autogluon (automl models)
"""


def train(x, y, method):
    print("point here", x.shape, y.shape)
    if method == "xgboost":
        model = xgb.XGBRegressor(verbosity=0, n_estimators=1000, learning_rate=0.1)
        model.fit(x, y, eval_metric="auc")
    elif method == "gbdt":
        method = GradientBoostingRegressor(random_state=42)
        model = method.fit(x, y)
    elif method == "random_forest":
        method = RandomForestRegressor(random_state=42, max_depth=2)
        model = method.fit(x, y)
    elif method == "decision_tree":
        model = DecisionTreeRegressor(random_state=42)
        model.fit(x, y)
    elif method == "catboost":
        model = CatBoostRegressor(
            iterations=2,
            learning_rate=1,
            depth=2,
            loss_function="RMSE",
            verbose=None,
            allow_writing_files=False,
        )
        model.fit(x, y)
    return model


def validate(x, y, model):
    y_pred = model.predict(x)
    evaluation_scores = metrics.print_metrics_regression(y, y_pred)
    return evaluation_scores


def test(x, y, model):
    y_pred = model.predict(x)
    evaluation_scores = metrics.print_metrics_regression(y, y_pred)
    return evaluation_scores


if __name__ == "__main__":
    data_path = "./dataset/tongji/processed_data/"

    x = pickle.load(open("./dataset/tongji/processed_data/x.pkl", "rb"))

    y = pickle.load(open("./dataset/tongji/processed_data/y.pkl", "rb"))

    x_lab_length = pickle.load(
        open("./dataset/tongji/processed_data/visits_length.pkl", "rb")
    )

    x = x.numpy()
    y = y.numpy()
    x_lab_length = x_lab_length.numpy()
    x_flat = []
    y_flat = []
    for i in range(len(x)):
        cur_visits = x_lab_length[i]
        for j in range(int(cur_visits)):
            x_flat.append(x[i][j])
            y_flat.append(y[i][j])
    x = np.array(x_flat)
    y = np.array(y_flat)
    y_outcome = y[:, 0]
    y_los = y[:, 1]
    model = train(x, y_los, "xgboost")
    val_evaluation_scores = validate(x, y_los, model)
