# IEEE 实验用分类头：全通道抑制 / 无抑制 / 随机 / 低相关 / 高相关 通道抑制
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn


class OrthoDenoiser(nn.Module):
    """
    双视图软正交投影去噪（与 ssc/classifier.py 中 OrthoDenoiser 一致）。

    对每个样本的特征向量 p（backbone），以 ssc_view1/view2 为噪声方向 n1/n2，
    先将两个 view 对齐到 backbone 语义空间，再分别计算正交投影系数并软去除：

        a1 = (p · n1) / (‖n1‖² + ε)
        a2 = (p · n2) / (‖n2‖² + ε)
        p_clean = p - α1·a1·n1 - α2·a2·n2
    """
    EPS = 1e-6

    def __init__(self, feat_dim: int):
        super().__init__()
        self.align1 = nn.Sequential(
            nn.Linear(feat_dim, feat_dim, bias=False), nn.LayerNorm(feat_dim)
        )
        self.align2 = nn.Sequential(
            nn.Linear(feat_dim, feat_dim, bias=False), nn.LayerNorm(feat_dim)
        )
        self.alpha1 = nn.Parameter(torch.full((1,), 0.1))
        self.alpha2 = nn.Parameter(torch.full((1,), 0.1))

    def forward(
        self, backbone_feat: torch.Tensor, ssc_view1: torch.Tensor, ssc_view2: torch.Tensor
    ) -> torch.Tensor:
        n1 = F.normalize(self.align1(ssc_view1), dim=-1)
        n2 = F.normalize(self.align2(ssc_view2), dim=-1)
        a1 = (backbone_feat * n1).sum(dim=-1, keepdim=True)
        a2 = (backbone_feat * n2).sum(dim=-1, keepdim=True)
        denoised = backbone_feat - self.alpha1 * a1 * n1 - self.alpha2 * a2 * n2
        return denoised


def _build_head(hidden: int, class_number: int) -> nn.Sequential:
    """各分类器共用的 hidden→class 分类头，保证输出 logits 维度一致。"""
    return nn.Sequential(
        nn.Linear(hidden, hidden),
        nn.LayerNorm(hidden),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(hidden, hidden // 2),
        nn.LayerNorm(hidden // 2),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(hidden // 2, class_number),
    )


def _build_proj(input_feature: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_feature, hidden),
        nn.LayerNorm(hidden),
        nn.GELU(),
        nn.Dropout(0.1),
    )


def _channel_pearson_correlation(
    ssc_view1: torch.Tensor, ssc_view2: torch.Tensor
) -> torch.Tensor:
    """逐通道 Pearson 相关：在 batch 维统计，返回 shape (D,)."""
    v1 = ssc_view1 - ssc_view1.mean(dim=0, keepdim=True)
    v2 = ssc_view2 - ssc_view2.mean(dim=0, keepdim=True)
    num = (v1 * v2).sum(dim=0)
    den = v1.norm(dim=0).clamp(min=1e-6) * v2.norm(dim=0).clamp(min=1e-6)
    return num / den


def _select_correlation_channel_mask(
    corr: torch.Tensor, suppress_ratio: float, select_low: bool
) -> torch.Tensor:
    """按通道相关性强弱选取 suppress_ratio 比例通道，返回 (D,) 0/1 mask。"""
    feat_dim = corr.numel()
    k = max(1, int(feat_dim * suppress_ratio))
    if select_low:
        idx = corr.topk(k, largest=False).indices
    else:
        idx = corr.topk(k, largest=True).indices
    mask = torch.zeros(feat_dim, device=corr.device, dtype=corr.dtype)
    mask[idx] = 1.0
    return mask


class _CorrelationSuppressClassifier(nn.Module):
    """基于 ssc_view1/view2 通道相关性的软正交抑制基类。"""

    def __init__(
        self,
        input_feature: int,
        class_number: int,
        suppress_ratio: float = 0.2,
        select_low: bool = True,
    ):
        super().__init__()
        if not 0.0 < suppress_ratio <= 1.0:
            raise ValueError("suppress_ratio 须在 (0, 1] 内")
        self.suppress_ratio = suppress_ratio
        self.select_low = select_low
        hidden = 512
        self.denoiser = OrthoDenoiser(input_feature)
        self.proj = _build_proj(input_feature, hidden)
        self.head = _build_head(hidden, class_number)

    def _mask_ssc_views(
        self, ssc_view1: torch.Tensor, ssc_view2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        corr = _channel_pearson_correlation(ssc_view1, ssc_view2)
        mask = _select_correlation_channel_mask(
            corr, self.suppress_ratio, select_low=self.select_low
        )
        return ssc_view1 * mask, ssc_view2 * mask

    def forward(
        self, ssc_view1: torch.Tensor, ssc_view2: torch.Tensor, backbone_feat: torch.Tensor
    ) -> torch.Tensor:
        ssc_v1, ssc_v2 = self._mask_ssc_views(ssc_view1, ssc_view2)
        denoised = self.denoiser(backbone_feat, ssc_v1, ssc_v2)
        return self.head(self.proj(denoised))


class EfficientClassifier(nn.Module):
    """全通道软正交抑制（与 ssc/classifier.py EfficientClassifier 一致）。"""

    def __init__(self, input_feature: int, class_number: int):
        super().__init__()
        hidden = 512
        self.denoiser = OrthoDenoiser(input_feature)
        self.proj = _build_proj(input_feature, hidden)
        self.head = _build_head(hidden, class_number)

    def forward(
        self, ssc_view1: torch.Tensor, ssc_view2: torch.Tensor, backbone_feat: torch.Tensor
    ) -> torch.Tensor:
        denoised = self.denoiser(backbone_feat, ssc_view1, ssc_view2)
        return self.head(self.proj(denoised))


class NoSuppressClassifier(nn.Module):
    """不做 SSC 抑制：直接以 backbone_feat 分类。"""

    def __init__(self, input_feature: int, class_number: int):
        super().__init__()
        hidden = 512
        self.proj = _build_proj(input_feature, hidden)
        self.head = _build_head(hidden, class_number)

    def forward(
        self, ssc_view1: torch.Tensor, ssc_view2: torch.Tensor, backbone_feat: torch.Tensor
    ) -> torch.Tensor:
        _ = (ssc_view1, ssc_view2)
        return self.head(self.proj(backbone_feat))


class RandomSuppressClassifier(nn.Module):
    """
    随机 20% 通道软正交抑制：ssc_view1/2 仅保留随机选取的 20% 通道，其余通道置 0 后送入 OrthoDenoiser。
    """

    def __init__(
        self,
        input_feature: int,
        class_number: int,
        suppress_ratio: float = 0.2,
    ):
        super().__init__()
        if not 0.0 < suppress_ratio <= 1.0:
            raise ValueError("suppress_ratio 须在 (0, 1] 内")
        self.suppress_ratio = suppress_ratio
        hidden = 512
        self.denoiser = OrthoDenoiser(input_feature)
        self.proj = _build_proj(input_feature, hidden)
        self.head = _build_head(hidden, class_number)

    def _sample_channel_mask(
        self, feat_dim: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        k = max(1, int(feat_dim * self.suppress_ratio))
        mask = torch.zeros(feat_dim, device=device, dtype=dtype)
        idx = torch.randperm(feat_dim, device=device)[:k]
        mask[idx] = 1.0
        return mask

    def _mask_ssc_views(
        self, ssc_view1: torch.Tensor, ssc_view2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feat_dim = ssc_view1.size(-1)
        mask = self._sample_channel_mask(feat_dim, ssc_view1.device, ssc_view1.dtype)
        return ssc_view1 * mask, ssc_view2 * mask

    def forward(
        self, ssc_view1: torch.Tensor, ssc_view2: torch.Tensor, backbone_feat: torch.Tensor
    ) -> torch.Tensor:
        ssc_v1, ssc_v2 = self._mask_ssc_views(ssc_view1, ssc_view2)
        denoised = self.denoiser(backbone_feat, ssc_v1, ssc_v2)
        return self.head(self.proj(denoised))


class LowCorClassifier(_CorrelationSuppressClassifier):
    """
    低相关 20% 通道软正交抑制：计算 ssc_view1/view2 逐通道 Pearson 相关，
    保留相关性最低的 20% 通道参与正交去除。
    """

    def __init__(
        self,
        input_feature: int,
        class_number: int,
        suppress_ratio: float = 0.2,
    ):
        super().__init__(
            input_feature, class_number, suppress_ratio=suppress_ratio, select_low=True
        )


class HighCorClassifier(_CorrelationSuppressClassifier):
    """
    高相关 20% 通道软正交抑制：计算 ssc_view1/view2 逐通道 Pearson 相关，
    保留相关性最高的 20% 通道参与正交去除。
    """

    def __init__(
        self,
        input_feature: int,
        class_number: int,
        suppress_ratio: float = 0.2,
    ):
        super().__init__(
            input_feature, class_number, suppress_ratio=suppress_ratio, select_low=False
        )
