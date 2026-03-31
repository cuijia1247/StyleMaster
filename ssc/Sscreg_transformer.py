import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from ssc.Backend import *

availableBackends = {'resnet18':resnet18, 'resnet34':resnet34, 'resnet50':resnet50, 'resnet101':resnet101,'resnet152':resnet152}

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
    input_size = 1024,
    output_size = 8192,
    depth_projector = 3,
    pretrained_backend=True,
    target_size = 224):

        super().__init__()
        
        
        # Load pretrained weights from local file if requested
        if backend == 'swin_base_patch4_window7_224':
            # Create model without pretrained weights first
            self.backend = timm.create_model(backend, pretrained=False, num_classes=0)
            pretrained_path = 'pretrainModels/swin_base_patch4_window7_224.pth'
            state_dict = torch.load(pretrained_path, map_location='cpu')
            self.backend.load_state_dict(state_dict, strict=False)
        elif backend == 'resnet50':
            self.backend = availableBackends[backend](pretrained=pretrained_backend)
        elif backend == 'vit_large_patch16_224':
            # Create model without pretrained weights first
            self.backend = timm.create_model(backend, pretrained=False, num_classes=0)
            pretrained_path = 'pretrainModels/vit_large_patch16_224.pth'
            state_dict = torch.load(pretrained_path, map_location='cpu')
            self.backend.load_state_dict(state_dict, strict=False)
            
        self.projector = MLP(input_size=input_size, output_size=output_size, depth=depth_projector)
        self.target_size = target_size

        # 冻结 backend 所有参数，只训练 projector
        for param in self.backend.parameters():
            param.requires_grad = False

    def forward(self, x):
        # 插值到目标尺寸 (默认 224x224)
        if x.shape[2] != self.target_size or x.shape[3] != self.target_size:
            x = F.interpolate(x, size=(self.target_size, self.target_size),
                            mode='bilinear', align_corners=False)

        # backend 已冻结，用 no_grad 避免保存中间激活，节省显存
        with torch.no_grad():
            x = self.backend(x)
        x = self.projector(x)
        return x

