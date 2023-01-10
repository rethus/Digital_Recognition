import torch
import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(                                             # input[1, 224, 224]
            nn.Conv2d(in_channels=1, out_channels=12, kernel_size=5, stride=3), # output[12, 74, 74]
            nn.MaxPool2d(kernel_size=2, stride=2)                               # output[12, 37, 37]
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=18, kernel_size=6),          # output[18, 32, 32]
            nn.MaxPool2d(kernel_size=2, stride=2)                               # output[18, 16, 16]
        )
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=18 * 16 * 16, out_features=1296)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=1296, out_features=256)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=256, out_features=10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 18 * 16 * 16)
        x = self.fc1(x)
        x=self.fc2(x)
        return self.fc3(x)
