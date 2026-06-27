# VICReg：Variance-Invariance-Covariance Regularization（参考 barlowtwins/barlow.py 结构）
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


class VICReg(nn.Module):
    """ResNet50 编码器 + 3 层 MLP 投影头，双视图前向返回 (z1, z2)。"""

    def __init__(self, in_features: int, proj_channels: int):
        super().__init__()
        self.encoder = resnet50(zero_init_residual=True)
        self.encoder.fc = nn.Identity()
        proj_layers: list[nn.Module] = []
        for i in range(3):
            if i == 0:
                proj_layers.append(nn.Linear(in_features, proj_channels, bias=False))
            else:
                proj_layers.append(nn.Linear(proj_channels, proj_channels, bias=False))
            if i < 2:
                proj_layers.append(nn.BatchNorm1d(proj_channels))
                proj_layers.append(nn.ReLU(inplace=True))
        self.proj = nn.Sequential(*proj_layers)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        z1 = self.proj(self.encoder(x1))
        z2 = self.proj(self.encoder(x2))
        return z1, z2


def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
    """取方阵非对角元素（VICReg 协方差正则用）。"""
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def vicreg_loss_fun(
    z1: torch.Tensor,
    z2: torch.Tensor,
    sim_coeff: float = 25.0,
    std_coeff: float = 25.0,
    cov_coeff: float = 1.0,
) -> torch.Tensor:
    """
    VICReg 三项损失：invariance(MSE) + variance(hinge) + covariance(off-diag)。
    默认系数与论文一致。
    """
    repr_loss = F.mse_loss(z1, z2)

    std_x = torch.sqrt(z1.var(dim=0) + 1e-4)
    std_y = torch.sqrt(z2.var(dim=0) + 1e-4)
    std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

    x = z1 - z1.mean(dim=0)
    y = z2 - z2.mean(dim=0)
    batch_size = x.size(0)
    cov_x = (x.T @ x) / (batch_size - 1)
    cov_y = (y.T @ y) / (batch_size - 1)
    cov_loss = (
        _off_diagonal(cov_x).pow(2).sum() / x.size(1)
        + _off_diagonal(cov_y).pow(2).sum() / y.size(1)
    )

    return sim_coeff * repr_loss + std_coeff * std_loss + cov_coeff * cov_loss
