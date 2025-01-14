import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import KFold
from densityratio.kmm import get_density_ratios
from model.custom_loss import custom_loss


# 训练模型的函数
def train_model_with_custom_loss(model, X_train, y_train, lr=0.01, epochs=100, n_splits=5):
    # 初始化KFold进行交叉验证

    global loss
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    total_loss = 0.0
    total_accuracy = 0.0
    for train_idx, val_idx in kf.split(X_train):
        # 根据索引分割训练集和验证集
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

        # 转换为tensor
        X_train_tensor = torch.tensor(X_train_fold, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_fold,  dtype=torch.long).view(-1)

        X_val_tensor = torch.tensor(X_val_fold, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val_fold,  dtype=torch.long).view(-1)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = custom_loss
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            loss = criterion(model, X_train_tensor, X_val_tensor, y_train_tensor, y_val_tensor)#, density_ratios)
            loss.backward()
            optimizer.step()

        # 在验证集上评估
        model.eval()
        with torch.no_grad():
            outputs = model(X_val_tensor)
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == y_val_tensor.view(-1)).float().mean()
            total_accuracy += accuracy.item()
            total_loss += loss.item()

    # 计算平均损失和准确率
    avg_loss = total_loss / n_splits
    avg_accuracy = total_accuracy / n_splits

    return avg_loss, avg_accuracy