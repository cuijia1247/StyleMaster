import torch
import torch.nn as nn
import torch.nn.functional as F


def format_cscae_feature_dims(
    num_ensembles: int,
    input_dim: int,
    hidden_dim: int,
    latent_dim: int,
    num_classes: int,
) -> str:
    """
    返回 CSCAE / SCAE 各段张量形状说明（K 路结构相同，参数不共享）。
    形状均按单样本特征向量 (dim,) 书写；batch 为 (N, dim)。
    """
    lines = [
        f"CSCAE 特征维度 (K={num_ensembles}, num_classes={num_classes}):",
        f"  输入 x: {input_dim}",
        f"  [每路 SCAE k=1..K] 编码: {input_dim} → (Linear) → {hidden_dim} → BN+ReLU → (Linear) → {latent_dim} 即 latent_k",
        f"  [每路 SCAE k] 解码: {latent_dim} → (Linear) → {hidden_dim} → BN+ReLU → (Linear) → {input_dim} 即 recon_k",
        f"  共识表示 consensus_latent = mean(latent_1..latent_K): {latent_dim}",
        f"  类中心 centers (可学习): ({num_classes}, {latent_dim})",
        f"  分类器 logits = Linear({latent_dim} → {num_classes}): {num_classes}",
    ]
    return "\n".join(lines)


class SCAE(nn.Module):
    """
    单个风格中心化自编码器 (Style Centralizing Auto-Encoder)
    """

    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(SCAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed


class CSCAE(nn.Module):
    """
    共识风格中心化自编码器 (Consensus Style Centralizing Auto-Encoder)
    集成 K 个 SCAE 并融合共识约束
    """

    def __init__(self, num_ensembles, input_dim, hidden_dim, latent_dim, num_classes):
        super(CSCAE, self).__init__()
        self.num_ensembles = num_ensembles
        self.num_classes = num_classes
        self.latent_dim = latent_dim

        self.autoencoders = nn.ModuleList(
            [SCAE(input_dim, hidden_dim, latent_dim) for _ in range(num_ensembles)]
        )

        # 可学习的风格类别中心
        self.centers = nn.Parameter(torch.randn(num_classes, latent_dim))

        self.classifier = nn.Linear(latent_dim, num_classes)

    def describe_feature_dims(self) -> str:
        """根据当前模块权重推断 input_dim / hidden_dim，并输出与 format_cscae_feature_dims 一致的说明。"""
        enc0 = self.autoencoders[0].encoder
        in_d = enc0[0].in_features
        hid = enc0[0].out_features
        return format_cscae_feature_dims(
            self.num_ensembles, in_d, hid, self.latent_dim, self.num_classes
        )

    def forward(self, x):
        latents = []
        reconstructions = []

        for ae in self.autoencoders:
            latent, recon = ae(x)
            latents.append(latent)
            reconstructions.append(recon)

        consensus_latent = torch.mean(torch.stack(latents), dim=0)
        logits = self.classifier(consensus_latent)

        return latents, reconstructions, consensus_latent, logits


class CSCAELoss(nn.Module):
    """联合损失：重构 + 类中心 + 共识 + 分类"""

    def __init__(self, lambda_recon=1.0, lambda_center=0.1, lambda_consensus=0.1):
        super(CSCAELoss, self).__init__()
        self.lambda_recon = lambda_recon
        self.lambda_center = lambda_center
        self.lambda_consensus = lambda_consensus

    def forward(self, x, labels, latents, reconstructions, consensus_latent, centers, logits):
        num_ensembles = len(latents)

        recon_loss = sum(F.mse_loss(recon, x) for recon in reconstructions) / num_ensembles

        batch_centers = centers[labels]
        center_loss = F.mse_loss(consensus_latent, batch_centers)

        consensus_loss = (
            sum(F.mse_loss(latent, consensus_latent) for latent in latents) / num_ensembles
        )

        ce_loss = F.cross_entropy(logits, labels)

        total_loss = (
            ce_loss
            + self.lambda_recon * recon_loss
            + self.lambda_center * center_loss
            + self.lambda_consensus * consensus_loss
        )

        return total_loss, {
            "ce_loss": ce_loss.item(),
            "recon_loss": recon_loss.item(),
            "center_loss": center_loss.item(),
            "consensus_loss": consensus_loss.item(),
        }
