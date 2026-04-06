import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tr
from PIL import Image

def criterion(x, y, lam_ortho=0.5, lam_var=1.0, lam_redundancy=0.1, epsilon=1e-3, device=torch.device('cuda')):
    """
    正交化损失：驱动 view1(x) 和 view2(y) 互相正交且非零，使两者成为独立的风格噪声方向。
      ortho_loss      : 最小化两视图的余弦相似度平方，驱动方向正交
      var_loss        : 各维度保持方差，防止任一视图退化为零向量
      redundancy_loss : 跨维度互相关矩阵非对角元惩罚，使 x 的各维与 y 的各维也去冗余
    """
    bs, emb = x.size()

    # 1. 正交化损失：余弦相似度平方，目标趋向 0（完全正交）
    x_n = F.normalize(x, dim=-1)
    y_n = F.normalize(y, dim=-1)
    cos_sim = (x_n * y_n).sum(dim=-1)          # (B,)
    ortho_loss = cos_sim.pow(2).mean()

    # 2. 方差损失：防止 x 或 y 坍缩为零向量
    std_x = torch.sqrt(x.var(dim=0) + epsilon)
    std_y = torch.sqrt(y.var(dim=0) + epsilon)
    var_loss = torch.mean(F.relu(1 - std_x)) + torch.mean(F.relu(1 - std_y))

    # 3. 跨维度冗余损失：惩罚 x 第 i 维与 y 第 j 维(i≠j)的互相关，按 D² 归一化防止高维爆炸
    xNorm = (x - x.mean(0)) / (x.std(0) + epsilon)
    yNorm = (y - y.mean(0)) / (y.std(0) + epsilon)
    cross_cor = (xNorm.T @ yNorm) / bs                          # (D, D)
    off_diag = ~torch.eye(emb, dtype=torch.bool, device=x.device)
    redundancy_loss = cross_cor.pow(2)[off_diag].sum() / (emb * emb)  # 除以 D² 归一化

    return lam_ortho * ortho_loss + lam_var * var_loss + lam_redundancy * redundancy_loss

def get_ssc_transforms(size, mean, std): #special for SSC training
    transformT = tr.Compose([
        transforms.ToTensor(),
        tr.RandomResizedCrop(size=size, scale=(0.4, 0.8), ratio=(3 / 4, 4 / 3)),  # scale 扩大范围，增强多样性
        tr.RandomRotation((-90, 90)),
        # tr.ColorJitter(),
        # tr.GaussianBlur(kernel_size=(23,23), sigma=(0.1, 2.0)),
        # tr.RandomGrayscale(p=0.2),
        tr.Normalize(mean, std),
        ])

    transformT1 = tr.Compose([
        transforms.ToTensor(),
        tr.RandomResizedCrop(size=size, scale=(0.4, 0.8), ratio=(3 / 4, 4 / 3)),  # scale 扩大范围，增强多样性
        tr.RandomRotation((-90, 90)),
        # tr.ColorJitter(),
        # tr.RandomGrayscale(p=0.2),
        # tr.GaussianBlur(kernel_size=(23,23), sigma=(0.1, 2.0)),
        tr.Normalize(mean, std),
        ])

    transformEvalT = tr.Compose([
        transforms.ToTensor(),
        tr.CenterCrop(size=size),
        tr.Normalize(mean, std),
        
    ])

    return transformT, transformT1, transformEvalT

def get_byol_transforms(size, mean, std):
    transformT = tr.Compose([
        transforms.ToTensor(),
        tr.RandomResizedCrop(size=size, scale=(0.2,0.8), ratio=(3 / 4, 4 / 3)),
        # tr.RandomRotation((-90, 90)),
        # tr.ColorJitter(),
        # tr.GaussianBlur(kernel_size=(23,23), sigma=(0.1, 2.0)),
        # tr.RandomGrayscale(p=0.2),
        tr.Normalize(mean, std),
        ])

    transformT1 = tr.Compose([
        transforms.ToTensor(),
        tr.RandomResizedCrop(size=size, scale=(0.2,0.8), ratio=(3 / 4, 4 / 3)),
        # tr.RandomRotation((-90, 90)),
        # tr.ColorJitter(),
        # tr.RandomGrayscale(p=0.2),
        # tr.GaussianBlur(kernel_size=(23,23), sigma=(0.1, 2.0)),
        tr.Normalize(mean, std),
        ])

    transformEvalT = tr.Compose([
        transforms.ToTensor(),
        tr.CenterCrop(size=size),
        tr.Normalize(mean, std),
        
    ])

    return transformT, transformT1, transformEvalT

from torchvision.transforms import transforms


class MultiViewDataInjector(object):
    def __init__(self, *args):
        self.transforms = args[0]
        self.random_flip = transforms.RandomHorizontalFlip()

    def __call__(self, sample, *with_consistent_flipping):
        if with_consistent_flipping:
            sample = self.random_flip(sample)
        output = [transform(sample) for transform in self.transforms]
        # image0 = Image.fromarray(output[0].numpy().transpose(1,2,0))
        # image0 = output[0].numpy().transpose(1,2,0)
        # image0.save('output0.png')
        return output