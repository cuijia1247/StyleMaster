"""
utils_add.py — 共性风格增强策略的 SSC 损失函数与数据变换

核心改变（相对于 utils.py）：
  原方案：正交化损失，驱动 view1 ⊥ view2 → 两视图成为独立噪声方向
  新方案：一致性对齐损失 + 跨视图有监督对比损失，驱动 view1 ≈ view2 提取公共风格，
          同时保证对齐的公共特征具有类间判别性。

损失设计分三项：
  - var_loss      : 防止特征坍缩为零向量
  - align_loss    : BarlowTwins 互相关，驱动两视图对齐公共风格
  - supcon_loss   : 跨视图有监督对比损失（Cross-View SupCon）：
                    对每个 view1 样本，以 view2 中同类样本为正、异类为负进行对比；
                    反向同理。只在跨视图间计算，不在同视图内部对比，
                    专注于"两个视图之间同类应相近、异类应相远"的目标。
"""
import torch
import torch.nn.functional as F
import torchvision.transforms as tr
from torchvision.transforms import transforms


def supcon_loss(x, y, labels, temperature=1.0):
    """
    跨视图有监督对比损失（Cross-View SupCon）。

    设计目标：
      对 view1 中每个样本 x_i，在 view2 的 B 个样本中：
        - label 相同的 y_j 为正样本 → 拉近
        - label 不同的 y_j 为负样本 → 推远
      反向（以 y_i 为 anchor，在 view1 中对比）同理，取双向均值。

    与原版 SupCon 的区别：
      - 原版：将 2B 个特征混合，同视图内部也参与对比，正样本包含同视图同类对
      - 本版：锚点只在一个视图，候选只在另一个视图，完全跨视图，不含同视图内部噪声
      - 本版 loss 期望值 ≈ log(B/n_pos_per_class)，量级更可控（B=128, C=13: ≈log(10)≈2.3）

    Args:
        x, y        : SSC 编码器输出，shape (B, D)，未归一化
        labels      : 类别标签，shape (B,)，long，0-indexed
        temperature : softmax 温度（默认 1.0，余弦相似度值域 [-1,1]，无溢出风险）

    返回：
        标量损失
    """
    B = labels.size(0)
    # L2 归一化：余弦相似度值域 [-1,1]，temperature=1.0 时 sim 值域同为 [-1,1]，无溢出
    xn = F.normalize(x, dim=1)                               # (B, D)
    yn = F.normalize(y, dim=1)                               # (B, D)

    # 跨视图相似度矩阵：sim_xy[i,j] = cos(x_i, y_j) / T
    sim_xy = xn @ yn.T / temperature                         # (B, B)

    # 正样本掩码：labels[i] == labels[j]
    labels_col = labels.unsqueeze(1)                          # (B, 1)
    mask_pos = (labels_col == labels_col.T)                   # (B, B)，含自身(i==j)

    # ── 方向1：以 x 为 anchor，在 y 中对比 ──────────────────────────────────────
    # 分子：对正样本位置取 log_softmax，自身(i==i)也是正样本（view2 中对应样本同类）
    log_prob_xy = F.log_softmax(sim_xy, dim=1)               # (B, B)
    n_pos_x = mask_pos.sum(dim=1).float().clamp(min=1)
    loss_x = -(torch.where(mask_pos, log_prob_xy, torch.zeros_like(log_prob_xy))
               .sum(dim=1) / n_pos_x)                        # (B,)

    # ── 方向2：以 y 为 anchor，在 x 中对比 ──────────────────────────────────────
    sim_yx = sim_xy.T                                         # (B, B)，cos(y_i, x_j)/T
    log_prob_yx = F.log_softmax(sim_yx, dim=1)               # (B, B)
    n_pos_y = mask_pos.T.sum(dim=1).float().clamp(min=1)
    loss_y = -(torch.where(mask_pos.T, log_prob_yx, torch.zeros_like(log_prob_yx))
               .sum(dim=1) / n_pos_y)                        # (B,)

    return (loss_x.mean() + loss_y.mean()) * 0.5


def criterion_align(x, y, labels=None, lam_var=1.0, lam_align=1.0,
                    lam_supcon=0.05, temperature=1.0, epsilon=1e-3,
                    device=torch.device('cuda')):
    """
    共性风格一致性损失 + 跨视图有监督对比损失。

    Args:
        x, y        : SSC 编码器对 view1/view2 的输出，shape (B, D)
        labels      : 类别标签 shape (B,)，传入时启用 SupCon 分量；为 None 时退化为纯对齐
        lam_var     : 方差损失权重，防止坍缩
        lam_align   : BarlowTwins 权重
        lam_supcon  : 跨视图 SupCon 权重，默认 0.05（辅助正则，不主导梯度方向）
        temperature : SupCon softmax 温度，默认 1.0（余弦相似度已归一化，无溢出风险）

    返回：
        总损失标量
    """
    bs, emb = x.size()

    # ── 1. 方差损失：防止 x 或 y 坍缩为零向量 ────────────────────────────────────
    std_x = torch.sqrt(x.var(dim=0) + epsilon)
    std_y = torch.sqrt(y.var(dim=0) + epsilon)
    var_loss = torch.mean(F.relu(1 - std_x)) + torch.mean(F.relu(1 - std_y))

    # ── 2. BarlowTwins 互相关损失（对角趋近 1 + 非对角趋近 0）────────────────────
    # clamp 确保分母不低于 epsilon，防止初始化阶段某维度方差为 0 时出现 NaN
    xNorm = (x - x.mean(0)) / x.std(0).clamp(min=epsilon)    # (B, D)
    yNorm = (y - y.mean(0)) / y.std(0).clamp(min=epsilon)    # (B, D)
    cross_cor = (xNorm.T @ yNorm) / bs                       # (D, D)

    on_diag  = torch.diagonal(cross_cor)
    on_loss  = (on_diag - 1).pow(2).mean()

    # 按实际非对角元数量 D*(D-1) 归一化，避免 D=1024 时分母达 10⁶ 导致惩罚近似为 0
    off_mask = ~torch.eye(emb, dtype=torch.bool, device=x.device)
    off_loss = cross_cor.pow(2)[off_mask].sum() / (emb * (emb - 1))

    align_loss = on_loss + 0.005 * off_loss

    total = lam_align * align_loss + lam_var * var_loss

    # ── 3. 有监督对比损失：保证对齐特征具有类间判别性 ────────────────────────────
    if labels is not None:
        sc_loss = supcon_loss(x, y, labels, temperature=temperature)
        total = total + lam_supcon * sc_loss

    return total


def get_ssc_transforms(size, mean, std):
    """数据增强变换（与 utils.py 保持一致，不做修改）"""
    transformT = tr.Compose([
        transforms.ToTensor(),
        tr.RandomResizedCrop(size=size, scale=(0.4, 0.8), ratio=(3/4, 4/3)),
        tr.RandomRotation((-90, 90)),
        tr.Normalize(mean, std),
    ])
    transformT1 = tr.Compose([
        transforms.ToTensor(),
        tr.RandomResizedCrop(size=size, scale=(0.4, 0.8), ratio=(3/4, 4/3)),
        tr.RandomRotation((-90, 90)),
        tr.Normalize(mean, std),
    ])
    transformEvalT = tr.Compose([
        transforms.ToTensor(),
        tr.CenterCrop(size=size),
        tr.Normalize(mean, std),
    ])
    return transformT, transformT1, transformEvalT


class MultiViewDataInjector(object):
    """双视图注入器（与 utils.py 完全相同）"""
    def __init__(self, *args):
        self.transforms = args[0]
        self.random_flip = transforms.RandomHorizontalFlip()

    def __call__(self, sample, *with_consistent_flipping):
        if with_consistent_flipping:
            sample = self.random_flip(sample)
        return [transform(sample) for transform in self.transforms]
