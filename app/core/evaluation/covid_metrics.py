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


def calculate_es(cur_out, cur_outcome_true, cur_los_true, thresholds):
    if cur_out > 0.5 and cur_outcome_true == 1:  # predict: 1, gt: 1
        return theta(los_true=cur_los_true, thresholds=thresholds, case="tp")
    elif cur_out <= 0.5 and cur_outcome_true == 1:  # predict: 0, gt: 1
        return theta(los_true=cur_los_true, thresholds=thresholds, case="fn")
    else:
        return theta(los_true=cur_los_true, thresholds=thresholds, case="tn|fp")


def early_prediction_outcome_metric(
    y_true, predictions, len_list, thresholds, verbose=0
):
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
    # print("len compare: ", num_records, len_list.sum())

    i = 0
    cur_patient_idx = 0
    while i < num_records:
        cur_patient_es_pred = []
        cur_patient_es_true = []
        outcome = 0
        for j in range(i, i + len_list[cur_patient_idx]):
            # print(j)
            cur_out = predictions[j]
            cur_gt = y_true[j, :]
            cur_outcome_true = cur_gt[0]
            outcome = cur_outcome_true
            # print(cur_outcome_true, end=", ")
            cur_los_true = cur_gt[1]
            cur_patient_es_pred.append(
                calculate_es(cur_out, cur_outcome_true, cur_los_true, thresholds)
            )
            cur_patient_es_true.append(
                calculate_es(
                    cur_outcome_true, cur_outcome_true, cur_los_true, thresholds
                )
            )
        cur_patient_es_pred = np.array(cur_patient_es_pred)
        cur_patient_es_true = np.array(cur_patient_es_true)

        # print("len:", j, cur_patient_es_pred.shape, cur_patient_es_true.shape)
        if outcome == 1:
            with np.errstate(divide="ignore", invalid="ignore"):
                c = np.sum(cur_patient_es_pred, axis=0) / np.sum(
                    cur_patient_es_true, axis=0
                )
                c[c == np.inf] = 1
                c[c < -1] = -1
                # if c[0] < -1:
                #     print(cur_patient_es_pred, cur_patient_es_true)
                c = np.nan_to_num(c, nan=1)
            metric.append(c)
        # print()
        i += len_list[cur_patient_idx]
        cur_patient_idx += 1
    # print("metric:", len(metric), len(metric[2]))
    result = np.array(metric)
    if verbose:
        print("Early Prediction Score:", result)
    # print(result.mean(axis=0), result)
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
