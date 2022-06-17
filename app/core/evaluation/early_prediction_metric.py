def early_prediction_outcome_metric(y_true, predictions, verbose=1):
    """
    > y_true: np.ndarray
      shape (num_patients, 2), 2 --> [outcome, los]
      eg: 2 records, they have outcome = 1, 0 separately
          then y_true = [[1, 5], [0, 3]]
          5 and 3 denotes their length of stay
    > predictions: np.ndarray
    return metric (type: List)

    note:
      - y/predictions are already flattened here
      - so we don't need to consider visits_length
    """
    metric = []
    num_records = len(y_true)
    for i in range(num_records):
        cur_out = predictions[i, :]
        cur_gt = y_true[i, :]
        if cur_out[0] > 0.5 and cur_gt[0] == 1:  # 预测死亡, 实际最终死亡
            # 判断实际的当前los
            if cur_gt[1] >= 2:  # 早期预测对
                metric.append(1)
            elif 2 > cur_gt[1] > 1:  # 中期预测对
                score = cur_gt[1] - 1
                metric.append(score)
            elif cur_gt[1] <= 1:  # 晚期预测对
                metric.append(0)
        elif cur_out[0] <= 0.5 and cur_gt[0] == 1:  # 预测存活, 实际最终死亡
            if cur_gt[1] >= 2:  # 早期预测错，没关系
                metric.append(0)
            elif 2 > cur_gt[1] > 1:  # 中期预测错
                score = cur_gt[1] - 2
                metric.append(score)
            elif cur_gt[1] <= 1:  # 晚期预测错
                metric.append(-1)
        else:
            # 其他case不关心，score为0即可
            metric.append(0)
    if verbose:
        print(metric)
    return metric


def early_prediction_los_metric(y_true, predictions, verbose=1):
    """
    > y_true: np.ndarray
      shape (num_patients, 2), 2 --> [outcome, los]
      eg: 2 records, they have outcome = 1, 0 separately
          then y_true = [[1, 5], [0, 3]]
          5 and 3 denotes their length of stay
    > predictions: np.ndarray
    return metric (type: List)

    note:
      - y/predictions are already flattened here
      - so we don't need to consider visits_length
    """
    pass
