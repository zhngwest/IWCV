import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


# This is another metrics that transfer the prob to label
def probability_to_class(y_prob, threshold=0.5):
    """

    parameters:
    y_prob (array-like): 预测概率。
    threshold (float): 将概率转换为类别的阈值。默认为 0.5。

    返回:
    array: 预测类别。
    the accuracy may be not right.
    """
    return (np.array(y_prob) >= threshold).astype(int)


def evaluate_model(y_true, y_prob, threshold=0.5, pos_label=1):
    """
    评估模型的性能，包括精度、召回率、F1分数和准确率，并计算综合的准确率指标。

    参数:
    y_true (array-like): 真实标签。
    y_prob (array-like): 预测概率。
    threshold (float): 将概率转换为类别的阈值。默认为 0.5。
    pos_label (int): 正类标签，默认为 1。

    返回:
    dict: 包含各项评估指标和综合准确率的字典。
    """
    y_pred = probability_to_class(y_prob, threshold)

    # precision = precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    # recall = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    # f1 = f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    # 计算综合准确率（这里简单使用各指标的平均值）
    # combined_accuracy = (precision + recall + f1 + accuracy) / 4

    # metrics = {
    #     'precision': precision,
    #     'recall': recall,
    #     'f1_score': f1,
    #     'accuracy': accuracy,
    #     'combined_accuracy': combined_accuracy
    # }

    return accuracy
