import torch
import torch.nn as nn
import os

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm library not available. Please install it with: pip install timm")

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
    backend = 'swin_base_patch4_window7_224',
    input_size = 2048,
    output_size = 8192,
    depth_projector = 3,
    pretrained_backend=False,
    pretrain_models_dir='pretrainModels'):

        super().__init__()
        
        if not TIMM_AVAILABLE:
            raise ImportError("timm library is required for Swin Transformer. Please install it with: pip install timm")
        
        # 检查并下载预训练模型
        self._ensure_pretrained_model(backend, pretrain_models_dir, pretrained_backend)
        
        # Use Swin Transformer as backend - 优先使用本地模型
        self.backend = timm.create_model(backend, pretrained=False, num_classes=0)  # num_classes=0 to get features only
        
        # 尝试加载本地预训练权重
        model_path = os.path.join(pretrain_models_dir, f"{backend}.pth")
        if os.path.exists(model_path):
            print(f"加载本地预训练模型: {model_path}")
            try:
                checkpoint = torch.load(model_path, map_location='cpu')
                if 'model_state_dict' in checkpoint:
                    # 新格式：包含model_state_dict的checkpoint
                    self.backend.load_state_dict(checkpoint['model_state_dict'], strict=False)
                else:
                    # 旧格式：直接是state_dict
                    self.backend.load_state_dict(checkpoint, strict=False)
                print("✅ 本地预训练模型加载成功")
            except Exception as e:
                print(f"⚠️  本地模型加载失败: {str(e)}")
                print("使用随机初始化权重")
        else:
            print(f"⚠️  本地模型文件不存在: {model_path}")
            print("使用随机初始化权重")
        
        # Get the feature dimension from the Swin Transformer
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            swin_features = self.backend(dummy_input)
            swin_feature_dim = swin_features.shape[1]
        
        # Add a linear layer to match the expected input_size for the projector
        self.feature_adapter = nn.Linear(swin_feature_dim, input_size)
        
        # Keep the same projector as original
        self.projector = MLP(input_size=input_size, output_size=output_size, depth=depth_projector)
    
    def _ensure_pretrained_model(self, backend, pretrain_models_dir, pretrained_backend):
        """检查本地预训练模型是否存在"""
        # 确保目录存在
        os.makedirs(pretrain_models_dir, exist_ok=True)
        
        # 检查模型文件是否存在
        model_path = os.path.join(pretrain_models_dir, f"{backend}.pth")
        
        if os.path.exists(model_path):
            print(f"✅ 本地预训练模型已存在: {model_path}")
        else:
            print(f"⚠️  本地预训练模型不存在: {model_path}")
            print("将使用随机初始化权重")
    
    def forward(self, x):
        # Extract features using Swin Transformer
        x = self.backend(x)
        # Adapt features to match expected input_size
        x = self.feature_adapter(x)
        # Project features using the same projector as original
        x = self.projector(x)
        return x
