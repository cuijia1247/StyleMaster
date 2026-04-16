from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 堆叠 DAE 单元数与 layer_dims 的关系：N 个单元 → layer_dims 长度为 N+1（输入 + N 个隐层输出维）
# 本仓库默认采用 **堆叠 2 个 DAE 单元**，即 layer_dims = [input_dim, h1, h2]（长度 3）。


def layer_dims_two_stacked_dae(input_dim: int, h1: int = 1024, h2: int = 512) -> list[int]:
    """
    构造「堆叠 2 个 DAE 单元」时的 layer_dims。
    第 1 个 DAE: input_dim→h1；第 2 个 DAE: h1→h2；分类头接在 h2 维上。
    """
    return [input_dim, h1, h2]


# ==========================================
# 1. 单层降噪自编码器 (Denoising Autoencoder)
# ==========================================
class DAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, corruption_rate=0.2, continuous_features: bool = False):
        super(DAE, self).__init__()
        self.corruption_rate = corruption_rate
        # True：主干 GAP 等实值特征，编码用 ReLU、解码线性输出 + MSE；False：像素 [0,1] + Sigmoid
        self.continuous_features = continuous_features

        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def get_corrupted_input(self, x):
        """注入 Masking Noise (随机置 0)"""
        if self.corruption_rate > 0:
            mask = (torch.rand_like(x) > self.corruption_rate).float()
            return x * mask
        return x

    def _encode_act(self, z: torch.Tensor) -> torch.Tensor:
        if self.continuous_features:
            return F.relu(z)
        return torch.sigmoid(z)

    def forward(self, x):
        """微调 / 推理：仅编码器路径（与逐层堆叠一致）"""
        return self._encode_act(self.encoder(x))

    def pretrain_forward(self, x):
        """无监督预训练：加噪 → 编解码；重建目标为未加噪的 h"""
        corrupted_x = self.get_corrupted_input(x)
        hidden = self._encode_act(self.encoder(corrupted_x))
        reconstructed = self.decoder(hidden)
        if not self.continuous_features:
            reconstructed = torch.sigmoid(reconstructed)
        return reconstructed, hidden

# ==========================================
# 2. 堆叠降噪自编码器分类网络 (Stacked DAE)
# ==========================================
class SDAEClassifier(nn.Module):
    def __init__(
        self,
        layer_dims,
        num_classes,
        corruption_rate=0.2,
        continuous_features: bool = False,
    ):
        """
        layer_dims: 各层宽度，长度为「堆叠单元数 + 1」。
        堆叠 2 个 DAE 时形如 [input_dim, h1, h2]（长度 3），与 layer_dims_two_stacked_dae 一致。
        更长列表表示更多堆叠层，例如 [784, 500, 500, 2000] 为 3 个 DAE 单元。
        """
        super(SDAEClassifier, self).__init__()
        self.num_layers = len(layer_dims) - 1
        self.continuous_features = continuous_features
        self.dae_layers = nn.ModuleList()

        for i in range(self.num_layers):
            dae = DAE(
                layer_dims[i],
                layer_dims[i + 1],
                corruption_rate,
                continuous_features=continuous_features,
            )
            self.dae_layers.append(dae)

        # 顶层分类器 (Linear + 交叉熵损失内部包含 Softmax)
        self.classifier = nn.Linear(layer_dims[-1], num_classes)

    def forward(self, x):
        """微调和推理阶段的前向传播"""
        for dae in self.dae_layers:
            x = dae(x) # 逐层提取特征 (仅使用编码器)
        logits = self.classifier(x)
        return logits

# ==========================================
# 3. 训练逻辑 (预训练 + 微调)
# ==========================================
def train_sdae(model, dataloader, epochs_pretrain=10, epochs_finetune=20, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # ------------------------------------------
    # 阶段一：逐层无监督预训练 (Layer-wise Pre-training)
    # ------------------------------------------
    print(">>> 开始逐层无监督预训练 (Pre-training)...")
    for i, dae in enumerate(model.dae_layers):
        print(f"正在预训练第 {i+1} 层 DAE...")
        optimizer = optim.Adam(dae.parameters(), lr=lr)
        criterion = nn.MSELoss() # 也可以使用 Binary Cross Entropy
        
        for epoch in range(epochs_pretrain):
            total_loss = 0
            for batch_x, _ in dataloader:
                batch_x = batch_x.view(batch_x.size(0), -1).to(device)
                
                # 获取当前层的输入：如果是第一层则是原始图像，否则是前一层的输出
                with torch.no_grad():
                    h = batch_x
                    for j in range(i):
                        h = model.dae_layers[j](h)
                
                optimizer.zero_grad()
                reconstructed, _ = dae.pretrain_forward(h)
                
                # 损失函数比较的是“重建输出”与“未加噪的隐藏表示 h”
                loss = criterion(reconstructed, h)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            print(f"  Epoch [{epoch+1}/{epochs_pretrain}], Loss: {total_loss/len(dataloader):.4f}")

    # ------------------------------------------
    # 阶段二：全局有监督微调 (Supervised Fine-tuning)
    # ------------------------------------------
    print("\n>>> 开始全局有监督微调 (Fine-tuning)...")
    # 此时优化器需要更新整个网络（所有编码器 + 分类头）
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs_finetune):
        total_loss = 0
        correct = 0
        total = 0
        for batch_x, labels in dataloader:
            batch_x = batch_x.view(batch_x.size(0), -1).to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs_finetune}], Loss: {total_loss/len(dataloader):.4f}, Acc: {acc:.2f}%")

    return model