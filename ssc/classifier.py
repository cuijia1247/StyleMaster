import logging
import time
import torch
import torch.nn.functional as F
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


class OrthoDenoiser(nn.Module):
    """
    双视图软正交投影去噪（图中公式的全局向量版本）。

    对每个样本的特征向量 p（backbone），以 ssc_view1/view2 为噪声方向 n1/n2，
    先将两个 view 对齐到 backbone 语义空间，再分别计算正交投影系数并软去除：

        a1 = (p · n1) / (‖n1‖² + ε)
        a2 = (p · n2) / (‖n2‖² + ε)
        p_clean = p - α1·a1·n1 - α2·a2·n2

    其中 α1, α2 为可学习标量，初始化为 0.1（温和去噪起点），
    训练过程中自适应调整每个 view 的去噪强度。
    """
    EPS = 1e-6

    def __init__(self, feat_dim):
        super().__init__()
        # 将两路 ssc_view 分别线性映射到 backbone 语义空间（无 bias，保留方向信息）
        self.align1 = nn.Sequential(nn.Linear(feat_dim, feat_dim, bias=False), nn.LayerNorm(feat_dim))
        self.align2 = nn.Sequential(nn.Linear(feat_dim, feat_dim, bias=False), nn.LayerNorm(feat_dim))
        # 两路可学习去噪强度系数（α1, α2），初始化为 0.1 以温和起步，避免早期破坏 backbone 语义
        self.alpha1 = nn.Parameter(torch.full((1,), 0.1))
        self.alpha2 = nn.Parameter(torch.full((1,), 0.1))

    def forward(self, backbone_feat, ssc_view1, ssc_view2):
        # align 后 L2 归一化：将噪声方向固定为单位向量，使 α 成为唯一的去噪强度控制量，
        # 同时避免 align 层通过幅值缩放绕过 α 造成过度去噪
        n1 = F.normalize(self.align1(ssc_view1), dim=-1)             # (B, D)，单位向量
        n2 = F.normalize(self.align2(ssc_view2), dim=-1)             # (B, D)，单位向量

        # ‖n‖² = 1，投影系数简化为点积，无需 EPS 分母，数值更稳定
        a1 = (backbone_feat * n1).sum(dim=-1, keepdim=True)          # (B, 1)
        a2 = (backbone_feat * n2).sum(dim=-1, keepdim=True)          # (B, 1)

        # 软正交去噪：p_clean = p - α1·a1·n1 - α2·a2·n2
        denoised = backbone_feat - self.alpha1 * a1 * n1 - self.alpha2 * a2 * n2
        return denoised


class EfficientClassifier(nn.Module):
    """
    软正交投影降噪 + 双路融合分类头：
      - backbone_feat : 预提取的 ViT 特征，维度 = input_feature
      - ssc_view1/2   : SSC 编码器对两个增强视图的输出，维度 = input_feature
    流程：
      1. OrthoDenoiser 联合两路 view 做软正交投影去噪，得到 p_clean
      2. p_clean 与 backbone_feat 分别投影到 512 维后拼接
      3. 3 层 MLP 输出分类 logit
    """
    def __init__(self, input_feature, class_number):
        super(EfficientClassifier, self).__init__()
        hidden = 512

        # 软正交投影去噪模块（联合处理 view1 和 view2）
        self.denoiser = OrthoDenoiser(input_feature)

        # 降噪后特征投影（Dropout 从 0.3 降至 0.1，加速收敛）
        self.denoise_proj = nn.Sequential(
            nn.Linear(input_feature, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        # 噪声残差分量投影：投影 (backbone - denoised) 而非原始 backbone，
        # 使两路输入真正互补（一路为去噪语义，一路为被去除的风格噪声）
        self.bb_proj = nn.Sequential(
            nn.Linear(input_feature, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        # 融合后分类头：1024 → 512 → 256 → class_num（Dropout 整体降低）
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(hidden // 2, class_number),
        )

    def forward(self, ssc_view1, ssc_view2, backbone_feat):
        # 联合软正交去噪：同时减去两路 view 的噪声方向分量
        denoised = self.denoiser(backbone_feat, ssc_view1, ssc_view2)

        # 噪声残差 = 被去除的风格方向分量，与去噪语义特征真正互补
        noise_residual = backbone_feat - denoised.detach()

        denoised_out = self.denoise_proj(denoised)          # 去噪语义特征 → 512
        bb_out       = self.bb_proj(noise_residual)         # 风格噪声残差 → 512
        fused        = torch.cat([denoised_out, bb_out], dim=-1)  # 拼接 → 1024
        return self.head(fused)                             # 分类 logit


class Classifier_Simple(nn.Module):
    """两层线性分类器：Linear → ReLU → Linear → ReLU，无 Dropout。
    隐藏层维度 = input_feature // 2。
    """
    def __init__(self, input_feature: int, class_number: int):
        super().__init__()
        hidden = input_feature // 2
        self.fc1 = nn.Linear(input_feature, hidden)
        self.fc2 = nn.Linear(hidden, class_number)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.fc2(self.act(self.fc1(x))))