# ViT-L/16 多数据集多次实验

## ViT-L/16 benchmark (Painting91, Pandora, ArtBench, FashionStyle14, Arch) (epochs=15, runs=5) — 2026-06-25 19:57:26

_data_base=`/mnt/codes/data/style/`_

_命令: `vit_l_16_test.py --benchmark_all`_

### Accuracy

| Dataset | num_classes | run1 | run2 | run3 | run4 | run5 | mean±std | data_root |
|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| Painting91 | 13 | 0.6492 | 0.6450 | 0.6597 | 0.6933 | 0.6471 | 0.6588±0.0201 | `/mnt/codes/data/style/Painting91` |
| Pandora | 12 | 0.5905 | 0.5911 | 0.5963 | 0.5957 | 0.5970 | 0.5941±0.0031 | `/mnt/codes/data/style/Pandora` |
| ArtBench | 10 | 0.5948 | 0.5946 | 0.5966 | 0.5966 | 0.5973 | 0.5960±0.0012 | `/mnt/codes/data/style/artbench-10-imagefolder-split` |
| FashionStyle14 | 14 | 0.7184 | 0.7285 | 0.7210 | 0.7251 | 0.7251 | 0.7236±0.0040 | `/mnt/codes/data/style/FashionStyle14` |
| Arch | 25 | 0.6433 | 0.6407 | 0.6417 | 0.6372 | 0.6397 | 0.6405±0.0023 | `/mnt/codes/data/style/Arch` |

### Macro-F1

| Dataset | num_classes | run1 | run2 | run3 | run4 | run5 | mean±std | data_root |
|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| Painting91 | 13 | 0.6481 | 0.6381 | 0.6546 | 0.6931 | 0.6429 | 0.6554±0.0220 | `/mnt/codes/data/style/Painting91` |
| Pandora | 12 | 0.5621 | 0.5626 | 0.5608 | 0.5672 | 0.5627 | 0.5631±0.0024 | `/mnt/codes/data/style/Pandora` |
| ArtBench | 10 | 0.5916 | 0.5920 | 0.5939 | 0.5936 | 0.5943 | 0.5931±0.0012 | `/mnt/codes/data/style/artbench-10-imagefolder-split` |
| FashionStyle14 | 14 | 0.7183 | 0.7298 | 0.7218 | 0.7264 | 0.7258 | 0.7244±0.0044 | `/mnt/codes/data/style/FashionStyle14` |
| Arch | 25 | 0.6380 | 0.6335 | 0.6327 | 0.6274 | 0.6312 | 0.6325±0.0038 | `/mnt/codes/data/style/Arch` |

### Weighted-F1

| Dataset | num_classes | run1 | run2 | run3 | run4 | run5 | mean±std | data_root |
|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| Painting91 | 13 | 0.6503 | 0.6454 | 0.6580 | 0.6939 | 0.6465 | 0.6588±0.0202 | `/mnt/codes/data/style/Painting91` |
| Pandora | 12 | 0.5669 | 0.5688 | 0.5666 | 0.5710 | 0.5678 | 0.5682±0.0018 | `/mnt/codes/data/style/Pandora` |
| ArtBench | 10 | 0.5916 | 0.5920 | 0.5939 | 0.5936 | 0.5943 | 0.5931±0.0012 | `/mnt/codes/data/style/artbench-10-imagefolder-split` |
| FashionStyle14 | 14 | 0.7163 | 0.7279 | 0.7199 | 0.7242 | 0.7238 | 0.7224±0.0044 | `/mnt/codes/data/style/FashionStyle14` |
| Arch | 25 | 0.6407 | 0.6381 | 0.6379 | 0.6337 | 0.6360 | 0.6373±0.0026 | `/mnt/codes/data/style/Arch` |

### Balanced Accuracy

| Dataset | num_classes | run1 | run2 | run3 | run4 | run5 | mean±std | data_root |
|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| Painting91 | 13 | 0.6410 | 0.6299 | 0.6486 | 0.6862 | 0.6347 | 0.6481±0.0225 | `/mnt/codes/data/style/Painting91` |
| Pandora | 12 | 0.5742 | 0.5723 | 0.5788 | 0.5804 | 0.5797 | 0.5771±0.0036 | `/mnt/codes/data/style/Pandora` |
| ArtBench | 10 | 0.5948 | 0.5946 | 0.5966 | 0.5966 | 0.5973 | 0.5960±0.0012 | `/mnt/codes/data/style/artbench-10-imagefolder-split` |
| FashionStyle14 | 14 | 0.7198 | 0.7300 | 0.7223 | 0.7266 | 0.7268 | 0.7251±0.0040 | `/mnt/codes/data/style/FashionStyle14` |
| Arch | 25 | 0.6357 | 0.6320 | 0.6313 | 0.6268 | 0.6296 | 0.6311±0.0033 | `/mnt/codes/data/style/Arch` |

## 汇总总表

| Dataset | num_classes | Accuracy | Macro-F1 | Weighted-F1 | Balanced Accuracy |
|---------|-------------|---------|---------|---------|---------|
| Painting91 | 13 | 0.6588±0.0201 | 0.6554±0.0220 | 0.6588±0.0202 | 0.6481±0.0225 |
| Pandora | 12 | 0.5941±0.0031 | 0.5631±0.0024 | 0.5682±0.0018 | 0.5771±0.0036 |
| ArtBench | 10 | 0.5960±0.0012 | 0.5931±0.0012 | 0.5931±0.0012 | 0.5960±0.0012 |
| FashionStyle14 | 14 | 0.7236±0.0040 | 0.7244±0.0044 | 0.7224±0.0044 | 0.7251±0.0040 |
| Arch | 25 | 0.6405±0.0023 | 0.6325±0.0038 | 0.6373±0.0026 | 0.6311±0.0033 |
