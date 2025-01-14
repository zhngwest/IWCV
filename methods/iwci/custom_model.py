# custom_model.py
# custom_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch import mode


# A new class rewrite the fit function use the rui
class CustomModel(nn.Module):
    def __init__(self, input_dim):
        super(CustomModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


# 定义自定义损失函数
def custom_loss(outputs, y_train, weights):
    # 确保输出和标签的形状匹配
    assert outputs.shape == y_train.shape, "Output and target shape mismatch"

    # 获取正样本和负样本的索引
    pos_indices = torch.where(y_train == 1)[0]
    neg_indices = torch.where(y_train == 0)[0]

    # 初始化损失
    total_loss = 0.0

    # 遍历每个正样本并计算损失
    for idx in pos_indices:
        total_loss += (outputs[idx] / len(pos_indices)) * weights[idx]
    # 遍历每个负样本并计算损失
    for idx in neg_indices:
        total_loss += ((-outputs[idx]) / len(neg_indices)) * weights[idx]

    # 返回损失值除以总样本数量
    return -total_loss


def train_custom_model(model, X_train, y_train, weights, lr=0.01, epochs=100):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = custom_loss

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    weights = torch.tensor(weights, dtype=torch.float32).view(-1, 1)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train, weights)
        loss.backward()
        optimizer.step()

    return model


def predict_custom_model(model, X):
    model.eval()
    X = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(X)
    return outputs.numpy()
