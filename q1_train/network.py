import torch
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as trans
import time

try:
    from hyperparameters import *
except:
    from .hyperparameters import *


def conv3x3(in_features, out_features): 
    return nn.Conv2d(in_features, out_features, kernel_size=3, padding=1)

class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        self.features = nn.Sequential(
            # 1.con1_1
            conv3x3(3, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 2.con1_2
            conv3x3(64, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 3.con2_1
            conv3x3(64, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 4.con2_2
            conv3x3(128, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 5.con3_1
            conv3x3(128, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 6.con3_2
            conv3x3(256, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 7.con3_3
            conv3x3(256, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 8.con3_4
            conv3x3(256, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 9.con4_1
            conv3x3(256, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 10.con4_2
            conv3x3(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 11.con4_3
            conv3x3(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 12.con4_4
            conv3x3(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 13.con5_1
            conv3x3(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 14.con5_2
            conv3x3(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 15.con5_3
            conv3x3(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 16.con5_4
            conv3x3(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
            )

        self.classifier = nn.Sequential(
            # 17.fc_1
            nn.Linear(512, 4096),
            nn.ReLU(),
            nn.Dropout(),
            # 18.fc_2
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            # 19.softmax
            nn.Linear(4096, 10),
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out