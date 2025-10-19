"""
Flexible SscReg with customizable backend output dimension
支持自定义backend输出维度的灵活版本
"""

import torch
import torch.nn as nn
from Backend import *

availableBackends = {
    'resnet18': resnet18, 
    'resnet34': resnet34, 
    'resnet50': resnet50, 
    'resnet101': resnet101,
    'resnet152': resnet152
}

# 不同backend的默认输出维度（avgpool后的特征维度）
BACKEND_OUTPUT_DIMS = {
    'resnet18': 512,
    'resnet34': 512,
    'resnet50': 2048,
    'resnet101': 2048,
    'resnet152': 2048,
}

class MLP(nn.Module):
    def __init__(self,
    input_size = 2048,
    output_size = 8192,
    depth = 3,
    ):  
        super().__init__()
        layers = []
        inp = input_size
        for d in range(depth):
            if d == depth - 1:
                layers.append(nn.Linear(inp, output_size))
            else:
                layers.extend([nn.Linear(inp, output_size), nn.BatchNorm1d(output_size), nn.ReLU(inplace=True)])
                inp = output_size
        self.layer = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layer(x)

class SscReg(nn.Module):
    def __init__(self,
    backend = 'resnet50',
    custom_backend_dim = None,  # 自定义backend输出维度（None表示使用默认维度）
    output_size = 8192,         # projector最终输出维度
    depth_projector = 3,        # projector深度
    pretrained_backend = False, # 是否使用预训练权重
    verbose = True):            # 是否打印信息
        """
        初始化SscReg模型
        
        Args:
            backend (str): backbone类型，可选 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
            custom_backend_dim (int, optional): 自定义backend输出维度
                - None: 使用backend的原生维度（resnet50为2048，resnet18为512）
                - int: 通过adapter层调整到指定维度
            output_size (int): projector的输出维度，用于对比学习
            depth_projector (int): projector的MLP深度
            pretrained_backend (bool): 是否加载ImageNet预训练权重
            verbose (bool): 是否打印模型配置信息
        
        Example:
            # 使用默认维度
            model = SscReg(backend='resnet50', output_size=8192)
            # ResNet50(2048) → Projector(8192)
            
            # 降维到512
            model = SscReg(backend='resnet50', custom_backend_dim=512, output_size=2048)
            # ResNet50(2048) → Adapter(512) → Projector(2048)
            
            # 升维到4096
            model = SscReg(backend='resnet50', custom_backend_dim=4096, output_size=8192)
            # ResNet50(2048) → Adapter(4096) → Projector(8192)
        """
        super().__init__()
        
        if backend not in availableBackends:
            raise ValueError(f"Backend '{backend}' not available. Choose from {list(availableBackends.keys())}")
        
        # 创建backend
        self.backend_name = backend
        self.backend = availableBackends[backend](pretrained=pretrained_backend)
        
        # 获取backend的原生输出维度
        self.native_backend_dim = BACKEND_OUTPUT_DIMS.get(backend, 2048)
        
        # 确定最终的backend输出维度
        if custom_backend_dim is None:
            # 使用原生维度
            self.backend_output_dim = self.native_backend_dim
            self.backend_adapter = None
            if verbose:
                print(f"[SscReg] Backend: {backend} (native dim: {self.native_backend_dim})")
        else:
            # 使用自定义维度，需要adapter
            self.backend_output_dim = custom_backend_dim
            if custom_backend_dim != self.native_backend_dim:
                self.backend_adapter = nn.Linear(self.native_backend_dim, custom_backend_dim)
                if verbose:
                    print(f"[SscReg] Backend: {backend} with adapter ({self.native_backend_dim} → {custom_backend_dim})")
            else:
                self.backend_adapter = None
                if verbose:
                    print(f"[SscReg] Backend: {backend} (custom dim equals native, no adapter needed)")
        
        # 创建projector
        self.projector = MLP(
            input_size=self.backend_output_dim, 
            output_size=output_size, 
            depth=depth_projector
        )
        
        self.output_size = output_size
        
        if verbose:
            print(f"[SscReg] Projector: {self.backend_output_dim} → {output_size} (depth={depth_projector})")
            print(f"[SscReg] Pretrained: {pretrained_backend}")
            print(f"[SscReg] Total params: {self.count_parameters() / 1e6:.2f}M")
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入图像 (Batch, 3, H, W)
        
        Returns:
            投影后的特征 (Batch, output_size)
        """
        # Step 1: Backend特征提取
        x = self.backend(x)  # (B, 3, H, W) → (B, native_backend_dim)
        
        # Step 2: Adapter维度调整（如果需要）
        if self.backend_adapter is not None:
            x = self.backend_adapter(x)  # (B, native_backend_dim) → (B, custom_backend_dim)
        
        # Step 3: Projector投影
        x = self.projector(x)  # (B, backend_output_dim) → (B, output_size)
        
        return x
    
    def count_parameters(self):
        """计算模型总参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_feature_dim(self):
        """返回backbone特征维度（adapter后）"""
        return self.backend_output_dim
    
    def get_output_dim(self):
        """返回projector输出维度"""
        return self.output_size


# 测试代码
if __name__ == "__main__":
    print("="*60)
    print("Testing SscReg with flexible backend dimension")
    print("="*60)
    
    # Test 1: 默认维度
    print("\n[Test 1] ResNet50 with default dimension")
    model1 = SscReg(backend='resnet50', output_size=8192, pretrained_backend=False)
    x1 = torch.rand(2, 3, 224, 224)
    y1 = model1(x1)
    print(f"Input: {x1.shape} → Output: {y1.shape}")
    
    # Test 2: 降维
    print("\n[Test 2] ResNet50 with dimension reduction (2048 → 512)")
    model2 = SscReg(backend='resnet50', custom_backend_dim=512, output_size=2048, pretrained_backend=False)
    x2 = torch.rand(2, 3, 224, 224)
    y2 = model2(x2)
    print(f"Input: {x2.shape} → Output: {y2.shape}")
    
    # Test 3: 升维
    print("\n[Test 3] ResNet50 with dimension expansion (2048 → 4096)")
    model3 = SscReg(backend='resnet50', custom_backend_dim=4096, output_size=8192, pretrained_backend=False)
    x3 = torch.rand(2, 3, 224, 224)
    y3 = model3(x3)
    print(f"Input: {x3.shape} → Output: {y3.shape}")
    
    # Test 4: ResNet18
    print("\n[Test 4] ResNet18 with default dimension")
    model4 = SscReg(backend='resnet18', output_size=2048, pretrained_backend=False)
    x4 = torch.rand(2, 3, 224, 224)
    y4 = model4(x4)
    print(f"Input: {x4.shape} → Output: {y4.shape}")
    
    # Test 5: ResNet18 + 自定义维度
    print("\n[Test 5] ResNet18 with custom dimension (512 → 1024)")
    model5 = SscReg(backend='resnet18', custom_backend_dim=1024, output_size=2048, pretrained_backend=False)
    x5 = torch.rand(2, 3, 224, 224)
    y5 = model5(x5)
    print(f"Input: {x5.shape} → Output: {y5.shape}")
    
    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)

