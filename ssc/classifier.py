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

class ResidualBlock(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, input_dim)
        self.linear2 = nn.Linear(input_dim, input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(0.1)
        self.activation = nn.GELU()
    
    def forward(self, x):
        residual = x
        out = self.linear1(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.layer_norm(out + residual)
        return out

class AdvancedClassifier(nn.Module):
    def __init__(self, input_feature, class_number):
        super().__init__()
        # 多头自注意力层
        self.attention = nn.MultiheadAttention(input_feature, num_heads=8, batch_first=True)
        # 残差连接
        self.residual_layers = nn.ModuleList([
            ResidualBlock(512) for _ in range(3)
        ])
        # 特征投影层
        self.feature_projection = nn.Linear(input_feature, 512)
        # 最终分类层
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, class_number)
        )
    
    def forward(self, x):
        # 输入维度处理: (batch_size, input_feature) -> (batch_size, 1, input_feature)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # 自注意力机制
        attn_output, _ = self.attention(x, x, x)
        
        # 移除序列维度: (batch_size, 1, input_feature) -> (batch_size, input_feature)
        attn_output = attn_output.squeeze(1)
        
        # 特征投影
        features = self.feature_projection(attn_output)
        
        # 残差连接
        for residual_layer in self.residual_layers:
            features = residual_layer(features)
        
        # 最终分类
        output = self.classifier(features)
        return output


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