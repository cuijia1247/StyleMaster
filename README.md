# SSC (SubStyleClassification done by CJ in Oct. 26th 2024)
## waiting for the new paper, released soon
Thanks for the contribution of code repository list
 * VicReg
 * SimCLR
 * Barlow Twin
 * ...

## Installation
 * CUDA (optional)
 * Pytorch
using the commands shown below to install the necessary packages:
'''
pip install -r requirements.txt
'''

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
'''
waiting for our new released paper citation
'''
