# SSC 
(SubStyleClassification done by CJ in Oct. 26th 2024)
## waiting for the new paper, released soon
Thanks for the contribution of code repository list
 * VicReg
 * SimCLR
 * Barlow Twin
 * ...

## Installation

**Requirements:**
- Python 3.8.19
- PyTorch 2.1.0
- CUDA (optional, for GPU acceleration)

**Install dependencies:**
```bash
pip install -r requirements.txt
```

## Dataset Setup

Organize your dataset in the following structure:
```
## Dataset setup
Dataset - test #the folder for test images with subfolders
        - train #the folder for train with subfoders
            - 1, 2, 3, ... #every class in a single sub-folder
Notice: the names of subfolders are class names
The demo dataset can be found in the following folder
'./data/DemoData/'


## Model

### ViT-based SSC Model (当前使用)
```python
from ssc.Sscreg_transformer import SscReg

# 使用ViT-large/16作为backbone
model = SscReg(
    backend='vit_large_patch16_224',
    input_size=1024,
    output_size=1024,
    depth_projector=3,
    pretrained_backend=False)
```

### ResNet-based SSC Model (旧版)
```python
from ssc.Sscreg import SscReg

model = SscReg(
    backend='resnet50',
    input_size=2048,
    output_size=256,
    depth_projector=3,
    pretrained_backend=False)
```

**模型特性：**
- 当前使用 ViT-large/16 作为backbone
- 基座模型输出维度：1024
- 支持本地预训练模型加载，完全离线运行



## Citation
```
waiting for our new released paper citation
```

在utils/文件夹下建立新的脚本imageFeatureExtraction.py, 对指定的数据库文件夹（/home/cuijia1247/Codes/SubStyleClassfication/data/Painting91/）读取所有jpg或png文件，通过指定的基线模型（vit_large_patch16_224）进行特征提取，提取指定维度的特征（1024）和文件名，并将其存储为pth文件，放入pretrainFeatures文件夹。
