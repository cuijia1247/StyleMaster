from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models

class ProjectionMLP(nn.Module):
    """
    两层多层感知机（MLP）投影头。
    用于将 ResNet 提取的高维特征映射到低维特征空间，以便进行对比学习或聚类。
    """
    def __init__(self, in_dim, hidden_dim=2048, out_dim=128):
        super(ProjectionMLP, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        # 通常在无监督学习中，会对投影后的特征进行 L2 归一化
        x = nn.functional.normalize(x, dim=1, p=2)
        return x


class ConCURLClassifier(nn.Module):
    """
    在冻结主干特征 h 上：ProjectionMLP → z（L2 归一化）→ 线性分类器。
    用于 concurl_train：h 来自预提取特征缓存，仅训练投影头 + 分类头。
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        hidden_mlp: int = 2048,
        projection_dim: int = 136,
    ):
        super().__init__()
        self.projector = ProjectionMLP(in_dim, hidden_mlp, projection_dim)
        self.classifier = nn.Linear(projection_dim, num_classes)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = self.projector(h)
        return self.classifier(z)


class ConCURLFeatureExtractor(nn.Module):
    """
    ConCURL 核心特征提取模块。
    包含一个骨干网络（如 ResNet50）和一个投影头（Projection MLP）。
    """
    def __init__(self, base_model='resnet50', hidden_mlp=2048, projection_dim=136):
        super(ConCURLFeatureExtractor, self).__init__()
        
        # 1. 加载基础的特征提取骨干网络（Base Encoder f_theta）
        if base_model == "resnet50":
            try:
                resnet = models.resnet50(weights=None)
            except TypeError:
                resnet = models.resnet50(pretrained=False)
            in_features = resnet.fc.in_features
        elif base_model == "resnet18":
            try:
                resnet = models.resnet18(weights=None)
            except TypeError:
                resnet = models.resnet18(pretrained=False)
            in_features = resnet.fc.in_features
        else:
            raise ValueError("Unsupported base model. Please choose 'resnet18' or 'resnet50'.")

        # 移除 ResNet 最后的分类层（fc），仅保留特征提取部分
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        
        # 2. 投影头（Projector g_theta）
        # 根据 ConCURL 论文设置，投影到特定的维度（如默认 136 维）
        self.projector = ProjectionMLP(in_dim=in_features, 
                                       hidden_dim=hidden_mlp, 
                                       out_dim=projection_dim)

    def forward(self, x):
        # 提取基础特征 (Batch Size, in_features, 1, 1)
        h = self.encoder(x)
        # 展平特征以输入到 MLP
        h = torch.flatten(h, 1)
        
        # 投影到低维特征空间
        z = self.projector(h)
        
        # 返回主干特征 h (用于下游任务评估) 和投影特征 z (用于计算 Consensus Loss)
        return h, z

# ==========================================
# 模块测试与使用示例
# ==========================================
if __name__ == "__main__":
    # 初始化 ConCURL 特征提取器（以 ResNet-50 为例）
    # 在开源库的 run.sh 中，通常使用 --hidden-mlp 2048 和 --projection-dim 136
    model = ConCURLFeatureExtractor(base_model='resnet50', hidden_mlp=2048, projection_dim=136)
    
    # 模拟输入：2张 3通道 160x160 大小的图像（ConCURL常用的裁剪尺寸）
    dummy_input = torch.randn(2, 3, 160, 160)
    
    # 前向传播
    representations, projections = model(dummy_input)
    
    print("模型结构初始化成功！")
    print(f"输入形状: {dummy_input.shape}")
    print(f"骨干网络提取的特征 (h) 形状: {representations.shape}") # 预期: [2, 2048]
    print(f"投影头输出的特征 (z) 形状: {projections.shape}")       # 预期: [2, 136]