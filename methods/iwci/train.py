# the process of train
import numpy as np
from sklearn.model_selection import KFold
from methods.custom_model import train_custom_model, predict_custom_model, CustomModel
from methods.evaluate_acc import evaluate_model
from evaluate.metrics_aucpy import auc_metric


def weighted_cross_validation(X_source, y_source, weights, input_dim, metric_fn, n_splits=5):
    # cross validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics = []
    metrics_another = []

    for train_index, val_index in kf.split(X_source):
        X_train, X_val = X_source[train_index], X_source[val_index]
        y_train, y_val = y_source[train_index], y_source[val_index]
        w_train = weights[train_index]

        model = CustomModel(input_dim=input_dim)
        model = train_custom_model(model, X_train, y_train, w_train)
        y_pred_proba = predict_custom_model(model, X_val)

        metric = auc_metric(y_val, y_pred_proba)
        metrics.append(metric)

        metric_another = evaluate_model(y_val, y_pred_proba)
        metrics_another.append(metric_another)

    mean_metric = np.mean(metrics)
    mean_metric_another = np.mean(metrics_another)

    return mean_metric, mean_metric_another


def train_final_model(X_source, y_source, weights, input_dim):
    model = CustomModel(input_dim=input_dim)
    model = train_custom_model(model, X_source, y_source, weights)
    return model
