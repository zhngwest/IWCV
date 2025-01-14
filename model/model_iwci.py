import torch
import torch.nn as nn
import torch.nn.functional as F

# replace the model before
class LeNet(nn.Module):
    def __init__(self, n_out):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(  # input_size=(3*32*32)
            nn.Conv2d(3, 6, kernel_size=5, padding=2),  # 输入通道改为3，保持输出通道为6
            nn.ReLU(),  # input_size=(6*32*32)
            nn.MaxPool2d(kernel_size=2, stride=2),  # output_size=(6*16*16)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5),  # input_size=(16*16*16)
            nn.ReLU(),  # output_size=(16*12*12)
            nn.MaxPool2d(2, 2)  # output_size=(16*6*6)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 6 * 6, 120),  # 调整输入特征数为 16*6*6
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, n_out)  # 最后一层设置为2个输出（2类别）

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0], -1)  # 展平张量
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)  # 输出两个类别的预估值
        return x
