import torch
from torch import nn

class CNNasza(nn.Module):
    def __init__(self, num_classes):
        
        super(CNNasza, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Conv2d(16, 120, kernel_size=5, stride=1)
        self.layer4 = nn.Sequential(
            nn.Linear(120, 84),
            nn.Linear(84, num_classes)
        )
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(-1, 120)
        out = self.layer4(out)
        return out