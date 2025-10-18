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
```python
from ssc.Sscreg import SscReg

model = SscReg(
    backend='resnet50',
    input_size=2048,
    output_size=256,
    depth_projector=3,
    pretrained_backend=False)
   
```



## Citation
```
waiting for our new released paper citation
```
