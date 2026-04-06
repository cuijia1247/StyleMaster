"""
utils_add.py — 共性风格增强策略的 SSC 损失函数与数据变换

核心改变（相对于 utils.py）：
  原方案：正交化损失，驱动 view1 ⊥ view2 → 两视图成为独立噪声方向
  新方案：一致性对齐损失 + 有监督对比损失，驱动 view1 ≈ view2 提取公共风格，
          同时保证对齐的公共特征具有类间判别性。

损失设计分三项：
  - var_loss      : 防止特征坍缩为零向量
  - align_loss    : BarlowTwins 互相关，驱动两视图对齐公共风格
  - supcon_loss   : 有监督对比损失，同类样本特征拉近、异类推远，
                    防止 BarlowTwins 把判别性方向对齐掉
"""
import torch
import torch.nn.functional as F
import torchvision.transforms as tr
from torchvision.transforms import transforms


def supcon_loss(x, y, labels, temperature=0.5, epsilon=1e-3):
    """
    有监督对比损失（SupConLoss，Khosla et al. 2020）。

    将 view1（x）和 view2（y）拼接为 2B 个特征，利用 labels 构造正样本掩码：
      - 同类的不同样本（含跨视图）互为正样本对
      - 异类样本为负样本

    Args:
        x, y        : SSC 编码器输出，shape (B, D)
        labels      : 类别标签，shape (B,)，long
        temperature : softmax 温度（默认 0.5，使用 L2 归一化后的余弦相似度，
                      数值已限制在 [-1,1]，较大温度避免 exp 溢出）

    返回：
        标量损失
    """
    B = x.size(0)
    # L2 归一化后余弦相似度范围为 [-1, 1]，避免高维内积数值过大导致 NaN
    feats = torch.cat([F.normalize(x, dim=1),
                       F.normalize(y, dim=1)], dim=0)         # (2B, D)

    # 拼接标签，shape (2B,)
    labels_2b = torch.cat([labels, labels], dim=0)            # (2B,)

    # 余弦相似度矩阵，除以温度；值域 [-1/T, 1/T]，温度 0.5 时最大值为 2，不会溢出
    sim = feats @ feats.T / temperature                       # (2B, 2B)

    # 去掉自相似（对角线置 -inf）
    mask_self = torch.eye(2 * B, dtype=torch.bool, device=x.device)
    sim = sim.masked_fill(mask_self, float('-inf'))

    # 正样本掩码：同类且非自身
    labels_2b = labels_2b.unsqueeze(1)                        # (2B, 1)
    mask_pos = (labels_2b == labels_2b.T) & ~mask_self        # (2B, 2B)

    # 过滤掉无正样本的行（batch 中某类只出现一次时），避免除零
    has_pos = mask_pos.any(dim=1)
    if not has_pos.any():
        return x.new_tensor(0.0)

    log_prob = F.log_softmax(sim, dim=1)                      # (2B, 2B)
    n_pos = mask_pos.sum(dim=1).float().clamp(min=1)
    # 用 torch.where 代替直接相乘，避免 mask_pos=False 处 -inf * 0 = NaN
    loss = -(torch.where(mask_pos, log_prob, torch.zeros_like(log_prob))
             .sum(dim=1) / n_pos)                             # (2B,)
    return loss[has_pos].mean()


def criterion_align(x, y, labels=None, lam_var=1.0, lam_align=1.0,
                    lam_supcon=0.5, temperature=0.5, epsilon=1e-3,
                    device=torch.device('cuda')):
    """
    共性风格一致性损失 + 有监督对比损失。

    Args:
        x, y        : SSC 编码器对 view1/view2 的输出，shape (B, D)
        labels      : 类别标签 shape (B,)，传入时启用 SupCon 分量；为 None 时退化为纯对齐
        lam_var     : 方差损失权重，防止坍缩
        lam_align   : BarlowTwins 权重
        lam_supcon  : 有监督对比损失权重（仅在 labels 不为 None 时生效）
        temperature : SupCon softmax 温度

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
        sc_loss = supcon_loss(x, y, labels, temperature=temperature, epsilon=epsilon)
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
