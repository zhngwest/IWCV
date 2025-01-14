import numpy as np
from sklearn.model_selection import KFold
from sklearn.base import clone
from methods.evaluate_acc import evaluate_model
from evaluate.metrics_aucpy import auc_metric


# implement the iwcv
def importance_weighted_cv(estimator, X_train, y_train, sample_weights, n_splits=5):
    """
    Perform importance weighted cross-validation.

    Parameters:
    - estimator: The machine learning model (must implement fit and predict methods).
    - X_train: Training feature matrix.
    - y_train: Training labels.
    - sample_weights: Importance weights for the training samples.
    - metric_fn: Function to evaluate model performance (e.g., roc_auc_score).
    - n_splits: Number of cross-validation folds.

    Returns:
    - mean_metric: Mean metric score over the cross-validation folds.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics = []
    metrics_another = []
    for train_index, val_index in kf.split(X_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
        sample_weights_fold = sample_weights[train_index]

        model = clone(estimator)
        print(np.isinf(sample_weights_fold).any())  # 检查 sample_weight
        model.fit(X_train_fold, y_train_fold, sample_weight=sample_weights_fold)
        y_pred = model.predict_proba(X_val_fold)[:, 1]

        metric = auc_metric(y_val_fold, y_pred)
        metrics.append(metric)

        metric_another = evaluate_model(y_val_fold, y_pred)
        metrics_another.append(metric_another)

    mean_metric = np.mean(metrics)
    mean_metric_another = np.mean(metrics_another)

    return mean_metric, mean_metric_another
