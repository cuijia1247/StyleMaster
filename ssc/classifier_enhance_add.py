"""
classifier_enhance_add.py — 共性风格增强融合分类器

核心假设（来自实验分析）：
  - 风格特征广泛存在于任意局部增强视图中，两视图的"交集"即为公共风格
  - SSC 编码器（新方案）学习两视图的共性，输出 ssc_view1 ≈ ssc_view2 ≈ 公共风格
  - 将提取到的公共风格特征以门控方式叠加到 backbone_feat 上，实现"风格增强"
  - 增强后的特征比原始 backbone_feat 包含更突出的风格信息，有利于分类

模块说明：
  StyleEnhancer   : 用双视图公共风格自适应增强 backbone 特征（门控加法）
  StyleEnhancer / SingleViewStyleEnhancer : 公共风格或单视图风格门控增强 backbone
  EfficientClassifier : 四路融合（各 256，cat→1024→head）
  RegionalWeightedPooling : 与 MCCFNet 一致；向量视作 (C,1,1)
  EfficientRWPClassifier : 四路分支同 EfficientClassifier；head 中 Dropout 换 RWP
"""
import torch
import torch.nn.functional as F
from torch import nn


class RegionalWeightedPooling(nn.Module):
    """
    区域加权池化 (RWP)：与 MCCFNet/mccfnet_train.py 一致。
    对特征图用 1×1 卷积产生空间权重，加权后全局平均池化。
    当输入为 (B, C, 1, 1)（向量视作 1×1 空间图）时，等价于通道维度的软门控后再聚合。
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.spatial_weight_conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.sigmoid(self.spatial_weight_conv(x))
        weighted_x = x * weights
        out = F.adaptive_avg_pool2d(weighted_x, (1, 1)).view(x.size(0), -1)
        return out


class RegionalWeightedPoolingVec(nn.Module):
    """对 (B, C) 向量应用 RegionalWeightedPooling：内部 reshape 为 (B, C, 1, 1)。"""
    def __init__(self, channels: int):
        super().__init__()
        self.rwp = RegionalWeightedPooling(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rwp(x.unsqueeze(-1).unsqueeze(-1))


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


class SingleViewStyleEnhancer(nn.Module):
    """
    单视图风格增强：与 StyleEnhancer 同结构，但风格方向仅来自单个 ssc_view（非双视图均值）。
    enhanced = backbone + alpha * sigmoid(gate_fc(backbone)) * normalize(align(ssc_view))
    """
    def __init__(self, feat_dim: int):
        super().__init__()
        self.align = nn.Sequential(
            nn.Linear(feat_dim, feat_dim, bias=False),
            nn.LayerNorm(feat_dim),
        )
        self.gate_fc = nn.Linear(feat_dim, feat_dim)
        self.alpha = nn.Parameter(torch.full((1,), 0.3))

    def forward(self, backbone_feat: torch.Tensor, ssc_view: torch.Tensor) -> torch.Tensor:
        style_dir = F.normalize(self.align(ssc_view), dim=-1)
        gate = torch.sigmoid(self.gate_fc(backbone_feat))
        return backbone_feat + self.alpha * gate * style_dir


class EfficientClassifier(nn.Module):
    """
    四路融合分类头（256×4=1024 → head）：

      路1：bb_proj(backbone_feat) — 原始全局语义，256
      路2：ssc_view1 对 backbone 的单视图增强后再投影，256
      路3：ssc_view2 对 backbone 的单视图增强后再投影，256
      路4：concat(ssc_view1, ssc_view2) 经 MLP → 256
    """
    def __init__(self, input_feature, class_number):
        super().__init__()

        self.enhance_v1 = SingleViewStyleEnhancer(input_feature)
        self.enhance_v2 = SingleViewStyleEnhancer(input_feature)

        # 路1：原始 backbone → 256（无 Dropout）
        self.bb_proj = nn.Sequential(
            nn.Linear(input_feature, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )
        # 路2 / 路3：增强后 backbone → 256
        self.v1_proj = nn.Sequential(
            nn.Linear(input_feature, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )
        self.v2_proj = nn.Sequential(
            nn.Linear(input_feature, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )
        # 路4：双视图拼接 → 256
        self.ssc_pair_mlp = nn.Sequential(
            nn.Linear(input_feature * 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )

        # 三层 Linear：1024 → 512 → 256 → class_number（无 Dropout）
        self.head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, class_number),
        )

    def forward(self, ssc_view1, ssc_view2, backbone_feat):
        feat_v1 = self.enhance_v1(backbone_feat, ssc_view1)
        feat_v2 = self.enhance_v2(backbone_feat, ssc_view2)

        out1 = self.bb_proj(backbone_feat)
        out2 = self.v1_proj(feat_v1)
        out3 = self.v2_proj(feat_v2)
        out4 = self.ssc_pair_mlp(torch.cat([ssc_view1, ssc_view2], dim=-1))

        fused = torch.cat([out1, out2, out3, out4], dim=-1)
        return self.head(fused)


class EfficientRWPClassifier(nn.Module):
    """
    与 EfficientClassifier 相同的四路分支与拼接；区别在 self.head：
    Dropout(0.05) 换为 RegionalWeightedPoolingVec(256)。
    """

    def __init__(self, input_feature: int, class_number: int):
        super().__init__()
        self.enhance_v1 = SingleViewStyleEnhancer(input_feature)
        self.enhance_v2 = SingleViewStyleEnhancer(input_feature)

        self.bb_proj = nn.Sequential(
            nn.Linear(input_feature, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.v1_proj = nn.Sequential(
            nn.Linear(input_feature, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.05),
        )
        self.v2_proj = nn.Sequential(
            nn.Linear(input_feature, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.05),
        )
        self.ssc_pair_mlp = nn.Sequential(
            nn.Linear(input_feature * 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.05),
        )
        self.head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            RegionalWeightedPoolingVec(256),
            nn.Linear(256, class_number),
        )

    def forward(self, ssc_view1, ssc_view2, backbone_feat):
        feat_v1 = self.enhance_v1(backbone_feat, ssc_view1)
        feat_v2 = self.enhance_v2(backbone_feat, ssc_view2)

        out1 = self.bb_proj(backbone_feat)
        out2 = self.v1_proj(feat_v1)
        out3 = self.v2_proj(feat_v2)
        out4 = self.ssc_pair_mlp(torch.cat([ssc_view1, ssc_view2], dim=-1))

        fused = torch.cat([out1, out2, out3, out4], dim=-1)
        return self.head(fused)
