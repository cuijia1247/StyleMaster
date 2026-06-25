# ResNet50 多数据集多次实验

## ResNet50 benchmark (Painting91, Pandora, ArtBench, FashionStyle14, Arch) (epochs=10, runs=5) — 2026-06-25 11:53:57

_data_base=`/mnt/codes/data/style/`_

_命令: `resnet50_test.py --benchmark_all`_

### Accuracy

| Dataset | num_classes | run1 | run2 | run3 | run4 | run5 | mean±std | data_root |
|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| Painting91 | 13 | 0.6050 | 0.6113 | 0.6261 | 0.6113 | 0.6176 | 0.6143±0.0079 | `/mnt/codes/data/style/Painting91` |
| Pandora | 12 | 0.5624 | 0.5585 | 0.5624 | 0.5572 | 0.5552 | 0.5591±0.0032 | `/mnt/codes/data/style/Pandora` |
| ArtBench | 10 | 0.5619 | 0.5621 | 0.5596 | 0.5588 | 0.5565 | 0.5598±0.0023 | `/mnt/codes/data/style/artbench-10-imagefolder-split` |
| FashionStyle14 | 14 | 0.6376 | 0.6384 | 0.6357 | 0.6312 | 0.6388 | 0.6363±0.0031 | `/mnt/codes/data/style/FashionStyle14` |
| Arch | 25 | 0.5981 | 0.5906 | 0.5991 | 0.5991 | 0.5866 | 0.5947±0.0058 | `/mnt/codes/data/style/Arch` |

### Macro-F1

| Dataset | num_classes | run1 | run2 | run3 | run4 | run5 | mean±std | data_root |
|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| Painting91 | 13 | 0.5776 | 0.5754 | 0.5975 | 0.5746 | 0.5852 | 0.5821±0.0096 | `/mnt/codes/data/style/Painting91` |
| Pandora | 12 | 0.5204 | 0.5059 | 0.5149 | 0.5228 | 0.5135 | 0.5155±0.0066 | `/mnt/codes/data/style/Pandora` |
| ArtBench | 10 | 0.5579 | 0.5586 | 0.5570 | 0.5570 | 0.5527 | 0.5566±0.0023 | `/mnt/codes/data/style/artbench-10-imagefolder-split` |
| FashionStyle14 | 14 | 0.6364 | 0.6375 | 0.6340 | 0.6315 | 0.6378 | 0.6354±0.0026 | `/mnt/codes/data/style/FashionStyle14` |
| Arch | 25 | 0.5836 | 0.5779 | 0.5827 | 0.5808 | 0.5717 | 0.5793±0.0048 | `/mnt/codes/data/style/Arch` |

### Weighted-F1

| Dataset | num_classes | run1 | run2 | run3 | run4 | run5 | mean±std | data_root |
|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| Painting91 | 13 | 0.5975 | 0.6004 | 0.6170 | 0.6019 | 0.6064 | 0.6046±0.0076 | `/mnt/codes/data/style/Painting91` |
| Pandora | 12 | 0.5346 | 0.5181 | 0.5318 | 0.5339 | 0.5253 | 0.5287±0.0070 | `/mnt/codes/data/style/Pandora` |
| ArtBench | 10 | 0.5579 | 0.5586 | 0.5570 | 0.5570 | 0.5527 | 0.5566±0.0023 | `/mnt/codes/data/style/artbench-10-imagefolder-split` |
| FashionStyle14 | 14 | 0.6336 | 0.6354 | 0.6320 | 0.6298 | 0.6358 | 0.6333±0.0025 | `/mnt/codes/data/style/FashionStyle14` |
| Arch | 25 | 0.5893 | 0.5832 | 0.5907 | 0.5886 | 0.5800 | 0.5863±0.0045 | `/mnt/codes/data/style/Arch` |

### Balanced Accuracy

| Dataset | num_classes | run1 | run2 | run3 | run4 | run5 | mean±std | data_root |
|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| Painting91 | 13 | 0.5809 | 0.5764 | 0.6017 | 0.5659 | 0.5756 | 0.5801±0.0133 | `/mnt/codes/data/style/Painting91` |
| Pandora | 12 | 0.5347 | 0.5236 | 0.5319 | 0.5320 | 0.5272 | 0.5299±0.0044 | `/mnt/codes/data/style/Pandora` |
| ArtBench | 10 | 0.5619 | 0.5621 | 0.5596 | 0.5588 | 0.5565 | 0.5598±0.0023 | `/mnt/codes/data/style/artbench-10-imagefolder-split` |
| FashionStyle14 | 14 | 0.6416 | 0.6394 | 0.6373 | 0.6341 | 0.6410 | 0.6387±0.0031 | `/mnt/codes/data/style/FashionStyle14` |
| Arch | 25 | 0.5878 | 0.5855 | 0.5904 | 0.5845 | 0.5761 | 0.5848±0.0054 | `/mnt/codes/data/style/Arch` |

## 汇总总表

| Dataset | num_classes | Accuracy | Macro-F1 | Weighted-F1 | Balanced Accuracy |
|---------|-------------|---------|---------|---------|---------|
| Painting91 | 13 | 0.6143±0.0079 | 0.5821±0.0096 | 0.6046±0.0076 | 0.5801±0.0133 |
| Pandora | 12 | 0.5591±0.0032 | 0.5155±0.0066 | 0.5287±0.0070 | 0.5299±0.0044 |
| ArtBench | 10 | 0.5598±0.0023 | 0.5566±0.0023 | 0.5566±0.0023 | 0.5598±0.0023 |
| FashionStyle14 | 14 | 0.6363±0.0031 | 0.6354±0.0026 | 0.6333±0.0025 | 0.6387±0.0031 |
| Arch | 25 | 0.5947±0.0058 | 0.5793±0.0048 | 0.5863±0.0045 | 0.5848±0.0054 |
