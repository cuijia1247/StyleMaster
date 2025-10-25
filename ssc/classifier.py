import logging
import time
import torch
from torch import nn
import torch.optim as optim
import torchvision.models as models
import torchvision.datasets as datasets
from torch.autograd import Variable
import numpy as np

class Classifier(nn.Module):
    def __init__(self, input_feature, class_number):
        super(Classifier, self).__init__()
        self.layer1 = nn.Linear(input_feature, 1024)
        self.layer2 = nn.Linear(1024, 256)
        self.layer3 = nn.Linear(256, class_number)
        self.dropout = nn.Dropout(0.5)
        self.activation_layer = nn.SiLU()


    def forward(self, input):
        hidden = self.layer1(input)
        hidden = self.activation_layer(hidden)
        if self.training == True:
            hidden = self.dropout(hidden)
        hidden = self.layer2(hidden)
        hidden = self.activation_layer(hidden)
        if self.training == True:
            hidden = self.dropout(hidden)
        out = self.layer3(hidden)
        # out = self.activation_layer(hidden)
        return out


class EfficientClassifier(nn.Module):
    def __init__(self, input_feature, class_number):
        super(EfficientClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_feature, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(128, class_number)
        )
    
    def forward(self, x):
        return self.classifier(x)