def early_prediction_outcome_metric(y_true, predictions, verbose=1):
    """
    > y_true: np.ndarray
    > predictions: np.ndarray
    return metric (type: List)
    """
    metric = []
    num_patient, max_visits_len, _ = y_true.shape
    for i in range(num_patient):
        for j in range(max_visits_len):
            cur_out = predictions[i, j, :]
            cur_gt = y_true[i, j, :]
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
    > predictions: np.ndarray
    return metric (type: List)
    """
    pass
