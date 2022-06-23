import torch
from torch import nn


class MTL(nn.Module):
    def __init__(self, task_num):
        super(WeightUncertaintyMTL, self).__init__()
        self.task_num = task_num
        self.alpha = nn.Parameter(torch.ones((task_num)))

    def forward(self, outcome_pred, outcome, los_pred, los):
        mse, bce = torch.nn.MSELoss(), torch.nn.BCELoss()
        loss0 = bce(outcome_pred, outcome)
        loss1 = mse(los_pred, los)
        return loss0 * self.alpha[0] + loss1 * self.alpha[1]


class WeightUncertaintyMTL(nn.Module):
    """
    Ref: Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics
    https://arxiv.org/abs/1705.07115
    """

    def __init__(self, task_num):
        super(WeightUncertaintyMTL, self).__init__()
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros((task_num)))

    def forward(self, outcome_pred, outcome, los_pred, los):
        mse, bce = torch.nn.MSELoss(), torch.nn.BCELoss()
        loss0 = bce(outcome_pred, outcome)
        loss1 = mse(los_pred, los)
        precision0 = torch.exp(-self.log_vars[0])
        loss0 = precision0 * loss0 + self.log_vars[0]
        precision1 = torch.exp(-self.log_vars[1])
        loss1 = precision1 * loss1 + self.log_vars[1]
        return loss0 + loss1


def predict_all_visits_bce_loss(y_pred, y_true, x_lab_length):
    """
    n-n prediction, mortality outcome
    """
    batch_size = len(y_true)
    loss = torch.nn.BCELoss()
    indices = torch.arange(batch_size, dtype=torch.int64)
    losses = 0
    for i in indices:
        losses += loss(
            y_pred[i][: x_lab_length[i].long()], y_true[i][: x_lab_length[i].long()]
        )
    return losses / batch_size


def predict_all_visits_mse_loss(y_pred, y_true, x_lab_length):
    """
    n-n prediction, length of stay
    """
    batch_size = len(y_true)
    loss = torch.nn.MSELoss()
    indices = torch.arange(batch_size, dtype=torch.int64)
    losses = 0
    for i in indices:
        losses += loss(
            y_pred[i][: x_lab_length[i].long()], y_true[i][: x_lab_length[i].long()]
        )
    return losses / batch_size


def get_multi_task_loss(
    y_outcome_pred, y_outcome_true, y_los_pred, y_los_true, x_lab_length
):
    batch_size = len(y_outcome_pred)
    loss = MTL(2)
    indices = torch.arange(batch_size, dtype=torch.int64)
    losses = 0
    for i in indices:
        losses += loss(
            y_outcome_pred[i][: x_lab_length[i].long()],
            y_outcome_true[i][: x_lab_length[i].long()],
            y_los_pred[i][: x_lab_length[i].long()],
            y_los_true[i][: x_lab_length[i].long()],
        )
    return losses / batch_size
