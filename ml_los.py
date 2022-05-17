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
from sklearn.model_selection import train_test_split
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


def train(x, y, x_lab_len, method):
    x_flat = []
    y_flat = []

    i = 0
    for i in range(len(x)):
        cur_visits = x_lab_len[i]
        for j in range(cur_visits):
            x_flat.append(x[i][j])
            y_flat.append(y[i][j])
    x = np.array(x_flat)
    y = np.array(y_flat)
    print("point here", x.shape, y.shape)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.8, random_state=42
    )
    if method == "xgboost":
        model = xgb.XGBRegressor(verbosity=0, n_estimators=100, learning_rate=0.1)
        model.fit(x_train, y_train, eval_metric="auc")
    # elif method == "logistic_regression":
    #     model = LogisticRegression(solver="liblinear")
    #     model.fit(x_train, y_train)
    elif method == "gbdt":
        method = GradientBoostingRegressor(random_state=42)
        model = method.fit(x_train, y_train)
    elif method == "random_forest":
        method = RandomForestRegressor(random_state=42, max_depth=2)
        model = method.fit(x_train, y_train)
    elif method == "decision_tree":
        model = DecisionTreeRegressor(random_state=42)
        model.fit(x_train, y_train)
    elif method == "catboost":
        model = CatBoostRegressor(
            iterations=2,
            learning_rate=1,
            depth=2,
            loss_function="RMSE",
            verbose=None,
            allow_writing_files=False,
        )
        model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    evaluation_scores = metrics.print_metrics_regression(y_test, y_pred)
    return evaluation_scores


if __name__ == "__main__":
    data_path = "./dataset/tongji/processed_data/"
    file_name = "./ckpt/gru.pth"

    batch_size = 64
    num_epochs = 100
    device = torch.device("cuda:0" if torch.cuda.is_available() == True else "cpu")
    # device = torch.device('cpu')
    print("available device: {}".format(device))

    x_lab = pickle.load(
        open("./dataset/tongji/processed_data/train_x_labtest.pkl", "rb")
    )
    x_lab = np.array(x_lab, dtype=object)
    x_lab = [torch.Tensor(_) for _ in x_lab]

    x_demo = pickle.load(
        open("./dataset/tongji/processed_data/train_x_demographic.pkl", "rb")
    )
    x_demo = np.array(x_demo)

    y_outcome = pickle.load(
        open("./dataset/tongji/processed_data/train_y_outcome.pkl", "rb")
    )
    y_outcome = np.array(y_outcome)

    y_los = pickle.load(open("./dataset/tongji/processed_data/train_y_LOS.pkl", "rb"))
    y_los = np.array(y_los, dtype=object)
    y_los = [torch.Tensor(_) for _ in y_los]

    x_lab_length = [len(_) for _ in x_lab]
    x_lab_length = np.array(x_lab_length)
    x_lab = torch.nn.utils.rnn.pad_sequence((x_lab), batch_first=True)
    y_los = torch.nn.utils.rnn.pad_sequence(y_los, batch_first=True)

    x_lab = np.array(x_lab)
    y_los = np.array(y_los)
    print(y_los)

    evaluation_scores = train(x_lab, y_los, x_lab_length, "random_forest")
