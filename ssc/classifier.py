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


class ViewProjector(nn.Module):
    """
    对单个 SSC view 输出做特征投影降噪：
      1. 将 ssc_view 投影到与 backbone 相同的语义空间，并通过 ReLU 过滤负分量
      2. 以 backbone 特征为"锚点"，用 Tanh 门控计算逐维度权重（值域 [-1,1]）
         - weight>0：抑制该维度，从 backbone 中减去对应投影分量
         - weight<0：增强该维度，向 backbone 中补充对应投影分量
         - weight=0：不做任何修改，完全保留 backbone 该维度信息
      3. denoised = backbone - weight × proj_view（受控残差）
    """
    def __init__(self, feat_dim):
        super().__init__()
        # 投影 ssc_view 到 backbone 语义空间；ReLU 过滤 ssc_view 中的负分量
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(),              # 只保留正向激活分量
        )
        # Tanh 门控：输出 [-1,1]，正值抑制、负值增强，比 Sigmoid 更灵活
        self.gate = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.Tanh(),
        )

    def forward(self, backbone_feat, ssc_view):
        proj_view  = self.proj(ssc_view)                              # ReLU 过滤后的投影
        gate_input = torch.cat([backbone_feat, proj_view], dim=-1)
        weight     = self.gate(gate_input)                            # 逐维度权重 ∈ [-1,1]
        denoised   = backbone_feat - weight * proj_view               # 受控残差
        return denoised


class EfficientClassifier(nn.Module):
    """
    特征投影降噪 + 双路融合分类头：
      - backbone_feat : 预提取的 ViT 特征，维度 = input_feature
      - ssc_view1/2   : SSC 编码器对两个增强视图的输出，维度 = input_feature
    流程：
      1. ViewProjector 对 view1/view2 分别做 ReLU 投影 + Sigmoid 门控降噪
      2. 可学习标量权重对两路降噪结果加权融合
      3. 融合特征与 backbone_feat 拼接，送入 3 层 MLP 分类
    """
    def __init__(self, input_feature, class_number):
        super(EfficientClassifier, self).__init__()
        hidden = 512

        # 两路 view 各自的投影降噪模块
        self.view1_proj = ViewProjector(input_feature)
        self.view2_proj = ViewProjector(input_feature)

        # 可学习的两路融合权重（Softmax 保证归一化）
        self.view_weight = nn.Parameter(torch.ones(2))

        # 降噪融合特征投影
        self.denoise_proj = nn.Sequential(
            nn.Linear(input_feature, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.3),
        )
        # Backbone 特征投影
        self.bb_proj = nn.Sequential(
            nn.Linear(input_feature, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.3),
        )
        # 融合后分类头
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(hidden // 2, class_number),
        )

    def forward(self, ssc_view1, ssc_view2, backbone_feat):
        denoised1 = self.view1_proj(backbone_feat, ssc_view1)
        denoised2 = self.view2_proj(backbone_feat, ssc_view2)

        # 可学习加权融合（Softmax 归一化，避免两路权重相互干扰）
        w = torch.softmax(self.view_weight, dim=0)
        denoised = w[0] * denoised1 + w[1] * denoised2

        denoised_out = self.denoise_proj(denoised)
        bb_out       = self.bb_proj(backbone_feat)
        fused        = torch.cat([denoised_out, bb_out], dim=-1)
        return self.head(fused)