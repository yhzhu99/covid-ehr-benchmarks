import math
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from autogluon.tabular import TabularPredictor
from catboost import CatBoostRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

"""
Tasks:
- mortality outcome
- los

Models:
- logistic regression (sklearn)
- random forest (sklearn)
- xgboost (xgboost)
- catboost (catboost)
- gbdt (sklearn)
- autogluon (automl models)
"""


def train(x, y, method):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.8, random_state=42
    )
    if method == "xgboost":
        model = xgb.XGBRegressor(verbosity=0, n_estimators=100, learning_rate=0.1)
        model.fit(x_train, y_train, eval_metric="auc")
    elif method == "logistic_regression":
        model = LogisticRegression(solver="liblinear")
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
    mape = format(mean_absolute_percentage_error(y_test, y_pred), ".2f")
    mae = format(mean_absolute_error(y_test, y_pred), ".2f")
    mse = format(mean_squared_error(y_test, y_pred), ".2f")
    rmse = format(math.sqrt(float(mse)), ".2f")
    r2 = format(r2_score(y_test, y_pred), ".2f")
