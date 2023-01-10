import torch
import torch.nn as nn


class BPNet(nn.Module):
    def __init__(self):
        super(BPNet, self).__init__()
        self.work = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(224 * 224, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        return self.work(x)
