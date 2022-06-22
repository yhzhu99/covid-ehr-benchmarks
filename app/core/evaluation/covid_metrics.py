import numpy as np
from sklearn import metrics as sklearn_metrics


def early_prediction_outcome_metric(y_true, predictions, verbose=1):
    """
    > predictions: np.ndarray
      shape (num_records, ) --> [outcome]
    > y_true: np.ndarray
      shape (num_records, 2), 2 --> [outcome, los]
      eg: 2 records, they have outcome = 1, 0 separately
          then y_true = [[1, 5], [0, 3]]
          5 and 3 denotes their length of stay
    return metric (type: float)

    note:
      - y/predictions are already flattened here
      - so we don't need to consider visits_length
    """
    metric = []
    num_records = len(predictions)
    for i in range(num_records):
        cur_out = predictions[i]
        cur_gt = y_true[i, :]
        if cur_out > 0.5 and cur_gt[0] == 1:  # predict: 1, gt: 1
            # check current los of ground truth (axis)
            if cur_gt[1] >= 2:  # predict right in early stage
                metric.append(1)
            elif 2 > cur_gt[1] > 1:  # predict right in medium stage
                score = cur_gt[1] - 1
                metric.append(score)
            elif cur_gt[1] <= 1:  # predict right in late stage
                metric.append(0)
        elif cur_out <= 0.5 and cur_gt[0] == 1:  # predict: 0, gt: 1
            if cur_gt[1] >= 2:  # predict wrong in early stage
                metric.append(0)
            elif 2 > cur_gt[1] > 1:  # predict wrong in medium stage
                score = cur_gt[1] - 2
                metric.append(score)
            elif cur_gt[1] <= 1:  # predict wrong in late stage
                metric.append(-1)
        else:
            # 其他case不关心，score为0即可
            metric.append(0)
    result = np.array(metric).mean()
    if verbose:
        print("Early Prediction Score:", result)
    return result


def sigma(los):
    """
    los = real los of patients (from this visit to the end)
    """
    if los >= 2:
        return 0
    elif 2 > los > 1:
        return 2 - los
    else:
        return 1


def multitask_los_metric(
    y_true,
    y_pred_outcome,
    y_pred_los,
    max_los=300,
    sigma_func=sigma,
    metrics_strategy="MAE",
    verbose=1,
):
    """
    > predictions: np.ndarray
      shape (num_records, ) --> [los]
    > y_true: np.ndarray
      shape (num_records, 2), 2 --> [outcome, los]
      eg: 2 records, they have outcome = 1, 0 separately
          then y_true = [[1, 5], [0, 3]]
          5 and 3 denotes their length of stay
    return metric (type: List)

    note:
      - y/predictions are already flattened here
      - so we don't need to consider visits_length
    """
    metric = 0
    y_true_outcome = y_true[:, 0]
    y_true_los = y_true[:, 1]
    if metrics_strategy == "MAE":
        metric += sklearn_metrics.median_absolute_error(y_true_los, y_pred_los)
    elif metrics_strategy == "MSE":
        metric += sklearn_metrics.mean_squared_error(y_true_los, y_pred_los)
    elif metrics_strategy == "MAPE":
        metric += sklearn_metrics.mean_absolute_percentage_error(y_true_los, y_pred_los)
    metric += np.mean(
        np.abs(y_true_outcome - y_pred_outcome)
        * max_los
        * np.array(list(map(lambda x: sigma_func(x), y_true_los)))
    )
    if verbose:
        print("LOS Score:", metric)
    return metric
