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
        
        # Use Swin Transformer as backend
        try:
            self.backend = timm.create_model(backend, pretrained=pretrained_backend, num_classes=0)  # num_classes=0 to get features only
        except Exception as e:
            if pretrained_backend:
                print(f"⚠️  无法加载预训练模型 {backend}: {str(e)}")
                print("尝试使用未预训练的模型...")
                self.backend = timm.create_model(backend, pretrained=False, num_classes=0)
            else:
                raise e
        
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
        """检查并下载预训练模型到指定目录"""
        if not pretrained_backend:
            return  # 如果不需要预训练模型，直接返回
        
        # 确保目录存在
        os.makedirs(pretrain_models_dir, exist_ok=True)
        
        # 检查模型文件是否存在
        model_path = os.path.join(pretrain_models_dir, f"{backend}.pth")
        
        if not os.path.exists(model_path):
            print(f"预训练模型 {backend} 不存在，开始下载...")
            print(f"保存路径: {os.path.abspath(model_path)}")
            
            try:
                # 创建模型并加载预训练权重
                model = timm.create_model(backend, pretrained=True, num_classes=0)
                
                # 保存模型
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'model_name': backend,
                    'num_features': model.num_features if hasattr(model, 'num_features') else model.head.in_features
                }, model_path)
                
                print(f"✅ 成功下载并保存: {model_path}")
                
                # 清理内存
                del model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except Exception as e:
                print(f"❌ 下载失败 {backend}: {str(e)}")
                print("将使用在线预训练权重...")
        else:
            print(f"✅ 预训练模型已存在: {model_path}")
    
    def forward(self, x):
        # Extract features using Swin Transformer
        x = self.backend(x)
        # Adapt features to match expected input_size
        x = self.feature_adapter(x)
        # Project features using the same projector as original
        x = self.projector(x)
        return x
