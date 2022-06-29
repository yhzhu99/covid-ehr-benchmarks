import numpy as np
from sklearn import metrics as sklearn_metrics


def theta(los_true, thresholds, case="tp"):
    """
    > case: str, "tp" or "fp"
    > outcome_pred: float, prediction of outcome
    > los_pred: float, prediction of los
    > los_true: float, true los
    return theta (type: float)
    """
    metric = []
    if case == "tp":
        for t in thresholds:
            if los_true >= t:  # predict right in early stage
                metric.append(1)
            else:
                metric.append(los_true / t)
        return np.array(metric)
    elif case == "fn":
        for t in thresholds:
            if los_true >= t:  # predict right in early stage
                metric.append(0)
            else:
                metric.append(los_true / t - 1)
        return np.array(metric)
    else:
        return np.zeros((len(thresholds),))


def early_prediction_outcome_metric(y_true, predictions, thresholds, verbose=0):
    """
    > predictions: np.ndarray
      shape (num_records, ) --> [outcome]
    > y_true: np.ndarray
      shape (num_records, 2), 2 --> [outcome, los]
      eg: 2 records, they have outcome = 1, 0 separately
          then y_true = [[1, 5], [0, 3]]
          5 and 3 denotes their length of stay
    > thresholds: List[float] e.g. [5,4,3.6044,3,2,1]]
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
        cur_outcome_true = cur_gt[0]
        cur_los_true = cur_gt[1]
        if cur_out > 0.5 and cur_outcome_true == 1:  # predict: 1, gt: 1
            metric.append(
                theta(los_true=cur_los_true, thresholds=thresholds, case="tp")
            )
        elif cur_out <= 0.5 and cur_outcome_true == 1:  # predict: 0, gt: 1
            metric.append(
                theta(los_true=cur_los_true, thresholds=thresholds, case="fn")
            )
        else:
            metric.append(
                theta(los_true=cur_los_true, thresholds=thresholds, case="tn|fp")
            )
    result = np.array(metric)
    if verbose:
        print("Early Prediction Score:", result)
    return result.mean(axis=0)


def calculate_epsilon(los_true, threshold, large_los):
    if los_true <= threshold:
        return 1
    else:
        return max(0, (los_true - large_los) / (threshold - large_los))


def sigma(los_pred, los_true, large_los, thresholds, case="true"):
    metric = []
    if case == "true":
        for t in thresholds:
            epsilon = calculate_epsilon(los_true, t, large_los)
            metric.append(epsilon * np.abs(los_pred - los_true))
        return np.array(metric)
    elif case == "false":
        for t in thresholds:
            epsilon = calculate_epsilon(los_true, t, large_los)
            metric.append(
                epsilon * (max(0, large_los - los_pred) + max(0, large_los - los_true))
            )
        return np.array(metric)
    else:
        raise ValueError("case must be 'true' or 'false'")


def calculate_outcome_prediction_result(outcome_pred, outcome_true):
    outcome_pred = 1 if outcome_pred > 0.5 else 0
    return "true" if outcome_pred == outcome_true else "false"


def multitask_los_metric(
    y_true,
    y_pred_outcome,
    y_pred_los,
    large_los,
    thresholds,
    verbose=0,
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
    metric = []
    num_records = len(y_pred_outcome)
    for i in range(num_records):
        cur_outcome_pred = y_pred_outcome[i]
        cur_los_pred = y_pred_los[i]
        cur_gt = y_true[i, :]
        cur_outcome_true = cur_gt[0]
        cur_los_true = cur_gt[1]
        prediction_result = calculate_outcome_prediction_result(
            cur_outcome_pred, cur_outcome_true
        )
        metric.append(
            sigma(
                cur_los_pred,
                cur_los_true,
                large_los,
                thresholds,
                case=prediction_result,
            )
        )
    result = np.array(metric)
    if verbose:
        print("Early Prediction Score:", result)
    return result.mean(axis=0)
