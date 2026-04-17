"""
SSC 编码器：冻结 DenseNet169 的 features（6 通道输入），全局池化后接可训练 MLP projector。
输出维度默认与 DenseNet169 分类器输入维一致（1664），便于与预提取 backbone 特征对齐。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import DenseNet169_Weights, densenet169

from ssc.Sscreg_transformer import MLP


class SscRegDensenet169(nn.Module):
    def __init__(
        self,
        input_size: int = 1664,
        output_size: int = 1664,
        depth_projector: int = 3,
        in_channels: int = 6,
        target_size: int = 224,
    ):
        super().__init__()
        net = densenet169(weights=DenseNet169_Weights.IMAGENET1K_V1)
        orig = net.features.conv0
        if in_channels != 3:
            net.features.conv0 = nn.Conv2d(
                in_channels,
                orig.out_channels,
                kernel_size=orig.kernel_size,
                stride=orig.stride,
                padding=orig.padding,
                bias=False,
            )
            with torch.no_grad():
                net.features.conv0.weight[:, :3] = orig.weight
                net.features.conv0.weight[:, 3:] = orig.weight.mean(dim=1, keepdim=True)

        self.features = net.features
        num_features = net.classifier.in_features
        if input_size != num_features:
            raise ValueError(
                f"input_size 应为 DenseNet169 池化维 {num_features}，收到 {input_size}"
            )

        self.projector = MLP(
            input_size=input_size, output_size=output_size, depth=depth_projector
        )
        self.target_size = target_size

        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[2] != self.target_size or x.shape[3] != self.target_size:
            x = F.interpolate(
                x,
                size=(self.target_size, self.target_size),
                mode="bilinear",
                align_corners=False,
            )
        with torch.no_grad():
            feat = self.features(x)
            feat = F.relu(feat, inplace=True)
            feat = F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)
        return self.projector(feat)
