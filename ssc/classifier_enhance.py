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
    双视图软正交投影去噪：ssc_view1/view2 各自经 align + L2 归一化后作为独立噪声方向，
    分别从 backbone_feat 中正交投影去除对应分量。

        n1 = L2_norm(align1(ssc_view1))
        n2 = L2_norm(align2(ssc_view2))
        a1 = backbone_feat · n1,  a2 = backbone_feat · n2
        p_clean = backbone_feat - α1·a1·n1 - α2·a2·n2

    align 后 L2 归一化将噪声方向固定为单位向量，使 α 成为唯一强度控制量，数值稳定。
    """
    def __init__(self, feat_dim):
        super().__init__()
        # 两路 ssc_view 分别映射到 backbone 语义空间（无 bias，保留方向信息）
        self.align1 = nn.Sequential(nn.Linear(feat_dim, feat_dim, bias=False), nn.LayerNorm(feat_dim))
        self.align2 = nn.Sequential(nn.Linear(feat_dim, feat_dim, bias=False), nn.LayerNorm(feat_dim))
        # 两路可学习去噪强度，初始化为 0.1（温和起步）
        self.alpha1 = nn.Parameter(torch.full((1,), 0.1))
        self.alpha2 = nn.Parameter(torch.full((1,), 0.1))

    def forward(self, backbone_feat, ssc_view1, ssc_view2):
        n1 = F.normalize(self.align1(ssc_view1), dim=-1)              # (B, D)，单位向量
        n2 = F.normalize(self.align2(ssc_view2), dim=-1)              # (B, D)，单位向量
        a1 = (backbone_feat * n1).sum(dim=-1, keepdim=True)           # (B, 1)
        a2 = (backbone_feat * n2).sum(dim=-1, keepdim=True)           # (B, 1)
        return backbone_feat - self.alpha1 * a1 * n1 - self.alpha2 * a2 * n2

    def forward_single(self, backbone_feat, ssc_view):
        """
        单视图去噪，对应代码变量关系如下（与下方三行实现一致）::

            n = F.normalize(self.align1(ssc_view), dim=-1)   # ‖n‖₂ = 1（最后一维）
            a = (backbone_feat * n).sum(dim=-1, keepdim=True)

            return backbone_feat - self.alpha1 * a * n

        即：在 `backbone_feat` 上减去沿 `n` 方向的投影标量 `a` 的 `self.alpha1` 倍。
        """
        n = F.normalize(self.align1(ssc_view), dim=-1)
        a = (backbone_feat * n).sum(dim=-1, keepdim=True)
        return backbone_feat - self.alpha1 * a * n

    def forward_single_v2(self, backbone_feat, ssc_view):
        """
        与 `forward_single` 相同结构，仅 `self.align1`→`self.align2`，`self.alpha1`→`self.alpha2`::

            n = F.normalize(self.align2(ssc_view), dim=-1)
            a = (backbone_feat * n).sum(dim=-1, keepdim=True)

            return backbone_feat - self.alpha2 * a * n
        """
        n = F.normalize(self.align2(ssc_view), dim=-1)
        a = (backbone_feat * n).sum(dim=-1, keepdim=True)
        return backbone_feat - self.alpha2 * a * n


class EfficientClassifier(nn.Module):
    """
    四路融合（256×4→1024→head）。默认 `forward`：路1 `bb_proj`；路2 `forward_single`（align1）；
    路3 `forward_single_v2`（align2）；路4 `residual_proj(bb−v1−v2)`。其它变体见类内注释。
    """
    def __init__(self, input_feature, class_number):
        super(EfficientClassifier, self).__init__()
        branch = 256  # 每路输出维度，四路拼接恰好为 1024

        # 路1：backbone 原始全局语义
        self.bb_proj = nn.Sequential(
            nn.Linear(input_feature, branch),
            nn.LayerNorm(branch),
            nn.GELU(),
            # nn.Dropout(0.1),
        )
        # 路2/3：backbone 减去单路 ssc_view 的残差，共享权重
        self.diff_proj = nn.Sequential(
            nn.Linear(input_feature, branch),
            nn.LayerNorm(branch),
            nn.GELU(),
            # nn.Dropout(0.1),
        )
        # 路4：ssc_view1 与 ssc_view2 的平均（双视图公共风格）
        self.ssc_avg_proj = nn.Sequential(
            nn.Linear(input_feature, branch),
            nn.LayerNorm(branch),
            nn.GELU(),
            # nn.Dropout(0.1),
        )
        # OrthoDenoiser：路2 用 forward_single(align1)，路3 用 forward_single_v2(align2)
        self.denoiser = OrthoDenoiser(input_feature)
        # 路2/3：单视图去噪后 D→256
        self.denoised_proj = nn.Sequential(
            nn.Linear(input_feature, branch),
            nn.LayerNorm(branch),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        # 路4：对 (backbone - ssc_v1 - ssc_v2) 投影，突出去掉两路风格后的残差语义
        self.residual_proj = nn.Sequential(
            nn.Linear(input_feature, branch),
            nn.LayerNorm(branch),
            nn.GELU(),
        )
        # 融合分类头：1024 → 512 → 256 → class_num
        self.head = nn.Sequential(
            nn.Linear(branch * 4, branch * 2),
            nn.LayerNorm(branch * 2),
            nn.GELU(),
            # nn.Dropout(0.15),

            nn.Linear(branch * 2, branch),
            nn.LayerNorm(branch),
            nn.GELU(),
            # nn.Dropout(0.1),

            nn.Linear(branch, class_number),
        )

    # def forward(self, ssc_view1, ssc_view2, backbone_feat):
    #     out1 = self.bb_proj(backbone_feat)                                       # 路1：原始语义 → 256
    #     out2 = self.denoised_proj(self.denoiser.forward_single(backbone_feat, ssc_view1))  # 路2：单视图去噪 → 256
    #     out3 = self.denoised_proj(self.denoiser.forward_single(backbone_feat, ssc_view2))  # 路3：单视图去噪 → 256（共享投影权重）
    #     out4 = self.ssc_avg_proj(0.5 * (ssc_view1 + ssc_view2))                 # 路4：双视图平均公共风格 → 256
    #     fused = torch.cat([out1, out2, out3, out4], dim=-1)                      # 拼接 → 1024
    #     return self.head(fused)                                                  # 分类 logit

    # ── 版本2：双视图降噪+DR（backbone-ssc_view），无 OrthoDenoiser ────────────────
    # def forward(self, ssc_view1, ssc_view2, backbone_feat):
    #     out1 = self.bb_proj(backbone_feat)                                     # 路1：原始语义 → 256
    #     out2 = self.diff_proj(backbone_feat - ssc_view1)                       # 路2：去除视图1风格的残差 → 256
    #     out3 = self.diff_proj(backbone_feat - ssc_view2)                       # 路3：去除视图2风格的残差 → 256（共享权重）
    #     out4 = self.denoised_proj(
    #         self.denoiser(backbone_feat, ssc_view1, ssc_view2)                 # 路4：双视图去噪语义 → 256
    #     )
    #     fused = torch.cat([out1, out2, out3, out4], dim=-1)                    # 拼接 → 1024
    #     return self.head(fused)
    # ── 版本1：单视图降噪+非噪声区补充（backbone-ssc_view），无 OrthoDenoiser ────────────────
    def forward(self, ssc_view1, ssc_view2, backbone_feat):
        # 路1：全局语义支路
        out1 = self.bb_proj(backbone_feat)
        # 路2：view1 噪声方向 align1 + alpha1（forward_single）
        out2 = self.denoised_proj(
            self.denoiser.forward_single(backbone_feat, ssc_view1)
        )
        # 路3：view2 噪声方向 align2 + alpha2（forward_single_v2）
        out3 = self.denoised_proj(
            self.denoiser.forward_single_v2(backbone_feat, ssc_view2)
        )
        # 路4：显式减去两路 SSC 表征后的残差，经 residual_proj 压到 256（互补路2/3 的「去噪」支路）
        out4 = self.residual_proj(backbone_feat - ssc_view1 - ssc_view2)
        fused = torch.cat([out1, out2, out3, out4], dim=-1)
        return self.head(fused)
