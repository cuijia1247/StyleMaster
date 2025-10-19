import torch
import torch.nn as nn
import timm

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
    backend = 'vit_large_patch16_224',
    input_size = 1024,
    output_size = 8192,
    depth_projector = 3,
    pretrained_backend=True):

        super().__init__()
        # Create model without pretrained weights first
        self.backend = timm.create_model(backend, pretrained=False, num_classes=0)
        
        # Load pretrained weights from local file if requested
        if pretrained_backend:
            pretrained_path = 'pretrainModels/vit_large_patch16_224.pth'
            state_dict = torch.load(pretrained_path, map_location='cpu')
            self.backend.load_state_dict(state_dict, strict=False)
        
        self.projector = MLP(input_size=input_size, output_size=output_size, depth=depth_projector)
    
    def forward(self, x):
        x = self.backend(x)
        x = self.projector(x)
        return x

