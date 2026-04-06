"""
classifier_enhance_add.py — 共性风格增强融合分类器

核心假设（来自实验分析）：
  - 风格特征广泛存在于任意局部增强视图中，两视图的"交集"即为公共风格
  - SSC 编码器（新方案）学习两视图的共性，输出 ssc_view1 ≈ ssc_view2 ≈ 公共风格
  - 将提取到的公共风格特征以门控方式叠加到 backbone_feat 上，实现"风格增强"
  - 增强后的特征比原始 backbone_feat 包含更突出的风格信息，有利于分类

模块说明：
  StyleEnhancer   : 用双视图公共风格自适应增强 backbone 特征（门控加法）
  StyleFusionClassifier : 三路融合分类头
      路1：backbone_feat（原始全局语义）
      路2：style_enhanced（风格增强后的 backbone）
      路3：ssc_common（双视图平均公共风格，直接投影）
"""
import torch
import torch.nn.functional as F
from torch import nn


class Classifier(nn.Module):
    """简单三层 MLP 分类头（保持与其他文件一致，供外部统一调用）"""
    def __init__(self, input_feature, class_number):
        super().__init__()
        self.layer1 = nn.Linear(input_feature, 1024)
        self.layer2 = nn.Linear(1024, 256)
        self.layer3 = nn.Linear(256, class_number)
        self.dropout = nn.Dropout(0.5)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.act(self.layer1(x))
        if self.training:
            x = self.dropout(x)
        x = self.act(self.layer2(x))
        if self.training:
            x = self.dropout(x)
        return self.layer3(x)


class StyleEnhancer(nn.Module):
    """
    公共风格增强模块：将双视图的公共风格以自适应门控方式叠加到 backbone_feat。

    流程：
        ssc_common = 0.5 * (ssc_view1 + ssc_view2)   # 双视图平均提取公共风格
        style_dir  = align(ssc_common)                # 映射到 backbone 语义空间（保留方向）
        gate       = sigmoid(gate_fc(backbone_feat))  # 逐维自适应门控系数 ∈ (0,1)
        enhanced   = backbone_feat + gate * style_dir # 门控叠加风格方向

    说明：
        - align 无 bias，L2 归一化后为单位方向向量，幅值由 gate 控制
        - gate 由 backbone_feat 驱动，使增强强度自适应于当前样本的语义状态
        - alpha 是全局可学习标量，控制整体风格增强幅度（初始值 0.3，温和起步）
        - 防坍缩：ssc_common 受 criterion_align 的 var_loss 约束，不会退化为零
    """
    def __init__(self, feat_dim):
        super().__init__()
        # 将公共风格映射到 backbone 语义空间（无 bias，保留方向信息）
        self.align = nn.Sequential(
            nn.Linear(feat_dim, feat_dim, bias=False),
            nn.LayerNorm(feat_dim),
        )
        # 逐维自适应门控：由 backbone_feat 决定每个维度的增强强度
        self.gate_fc = nn.Linear(feat_dim, feat_dim)
        # 全局增强幅度，初始 0.3（温和起步，避免早期破坏 backbone 语义）
        self.alpha = nn.Parameter(torch.full((1,), 0.3))

    def forward(self, backbone_feat, ssc_view1, ssc_view2):
        # 公共风格 = 双视图平均（两视图对齐后的共性）
        ssc_common = 0.5 * (ssc_view1 + ssc_view2)               # (B, D)
        # 归一化风格方向为单位向量，使 alpha/gate 成为唯一幅值控制量
        style_dir = F.normalize(self.align(ssc_common), dim=-1)   # (B, D)
        # 门控：由 backbone 语义决定各维的增强接受度
        gate = torch.sigmoid(self.gate_fc(backbone_feat))         # (B, D) ∈ (0,1)
        # 增强：在 backbone 上叠加风格方向
        enhanced = backbone_feat + self.alpha * gate * style_dir  # (B, D)
        return enhanced, ssc_common


class EfficientClassifier(nn.Module):
    """
    风格增强三路融合分类头（1024 维）：

      路1 bb_proj(backbone_feat)         : 原始 ViT 全局语义，256 维
      路2 enhanced_proj(style_enhanced)  : 风格增强后的 backbone（门控叠加公共风格），512 维
      路3 ssc_proj(ssc_common)           : 双视图均值公共风格 0.5*(v1+v2)，256 维

    三路 cat → 1024 → MLP head → class logit
    """
    def __init__(self, input_feature, class_number):
        super().__init__()

        # 公共风格增强模块（核心）
        self.enhancer = StyleEnhancer(input_feature)

        # 路1：原始 backbone 语义 → 256
        self.bb_proj = nn.Sequential(
            nn.Linear(input_feature, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        # 路2：风格增强后的 backbone → 512
        self.enhanced_proj = nn.Sequential(
            nn.Linear(input_feature, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        # 路3：双视图均值公共风格 → 256
        self.ssc_proj = nn.Sequential(
            nn.Linear(input_feature, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        # 融合分类头：1024 → 512 → 256 → class_num
        self.head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.15),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(256, class_number),
        )

    def forward(self, ssc_view1, ssc_view2, backbone_feat):
        # StyleEnhancer：提取公共风格并叠加到 backbone（丢弃返回的 ssc_common）
        style_enhanced, _ = self.enhancer(backbone_feat, ssc_view1, ssc_view2)
        # 路3 直接取双视图均值作为公共风格
        ssc_common = 0.5 * (ssc_view1 + ssc_view2)

        out1 = self.bb_proj(backbone_feat)        # 路1：原始语义 → 256
        out2 = self.enhanced_proj(style_enhanced)  # 路2：风格增强语义 → 512
        out3 = self.ssc_proj(ssc_common)           # 路3：公共风格均值 → 256

        fused = torch.cat([out1, out2, out3], dim=-1)  # 拼接 → 1024
        return self.head(fused)                        # 分类 logit
