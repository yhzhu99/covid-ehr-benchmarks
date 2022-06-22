import numpy as np
import torch


def flatten_dataset(x, y, indices, visits_length, case="los"):
    x_flat = []
    y_flat = []
    for i in indices:
        for v in range(visits_length[i]):
            x_flat.append(x[i][v])
            # if case == "los":
            #     y_flat.append(y[i][v][1])
            # elif case == "outcome":
            #     y_flat.append(y[i][v][0])
            y_flat.append(y[i][v].tolist())
    return np.array(x_flat), np.array(y_flat)


def numpy_dataset(x, y, x_lab_length):
    x = x.numpy()
    y = y.numpy()
    x_lab_length = x_lab_length.numpy()
    y_los = y[:, :, 1]
    y_outcome = y[:, 0, 0]
    return x, y_outcome, y_los, x_lab_length
