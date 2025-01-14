import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F

def evaluate_on_test_set(net, test_loader, device='cuda'):
    """
    在测试集上评估模型的表现，计算 Accuracy 和 AUC。

    Args:
        net (torch.nn.Module): 训练好的模型。
        test_loader (DataLoader): 测试集数据加载器。
        device (str): 设备 ('cuda' 或 'cpu')，默认使用 'cuda'。

    Returns:
        tuple: 返回 accuracy 和 AUC。
    """
    net.eval()  # 切换到评估模式
    all_labels = []
    all_preds = []

    # 在测试集上进行推理
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = net(images)

            # 预测类标签
            _, predicted = torch.max(outputs, 1)

            # 计算准确率
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 收集所有标签和预测值，用于计算 AUC
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(F.softmax(outputs, dim=1)[:, 1].cpu().numpy())  # 获取正类的概率

    # 计算准确率
    accuracy = correct / total

    # 计算 AUC
    auc = roc_auc_score(all_labels, all_preds)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test AUC: {auc:.4f}")

    return accuracy, auc
