import torch
from torch import nn


class Classifier(nn.Module):
    def __init__(self, input_feature, class_number):
        super(Classifier, self).__init__()
        self.layer1 = nn.Linear(input_feature, 1024)
        self.layer2 = nn.Linear(1024, 256)
        self.layer3 = nn.Linear(256, class_number)
        self.dropout = nn.Dropout(0.5)
        self.activation_layer = nn.SiLU()

    def forward(self, input):
        hidden = self.activation_layer(self.layer1(input))
        if self.training:
            hidden = self.dropout(hidden)
        hidden = self.activation_layer(self.layer2(hidden))
        if self.training:
            hidden = self.dropout(hidden)
        return self.layer3(hidden)


class EfficientClassifier(nn.Module):
    """
    简化直接融合分类头。

    forward 输入与 classifier_enhance.py 保持一致：
        ssc_view1, ssc_view2, backbone_feat  （三路维度相同，均为 input_feature）

    融合策略：
        fused = backbone_feat - 0.5 * ssc_view1 - 0.5 * ssc_view2
    直接将 SSC 双视图作为风格噪声方向，以固定系数从 backbone 中减去，
    无需任何 MLP 映射，特征维度保持不变，送入分类头。
    """
    def __init__(self, input_feature, class_number):
        super(EfficientClassifier, self).__init__()
        hidden = 512
        # 分类头：input_feature → 512 → 256 → class_number
        self.head = nn.Sequential(
            nn.Linear(input_feature, hidden),
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
        # 直接减去双视图的风格分量，无需 MLP 映射
        fused1 = backbone_feat - 0.5 * ssc_view1
        fused2 = backbone_feat - 0.5 * ssc_view2
        fused = fused1 + fused2
        return self.head(fused)
