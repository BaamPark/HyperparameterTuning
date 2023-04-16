import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, regularized=True, prob=0.3):
        super(LeNet, self).__init__()
        self.regularized = regularized

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 120, kernel_size=5),
            nn.ReLU()
        )

        if regularized:
            self.fc = nn.Sequential(
                nn.Linear(120, 84),
                nn.BatchNorm1d(84),
                nn.ReLU(),
                nn.Dropout(prob),
                nn.Linear(84, 10),
                nn.LogSoftmax(dim=-1)
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(120, 84),
                nn.ReLU(),
                nn.Linear(84, 10),
                nn.LogSoftmax(dim=-1)
            )
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out

