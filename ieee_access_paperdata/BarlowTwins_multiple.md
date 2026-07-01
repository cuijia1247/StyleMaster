# Barlow Twins 多数据集多次实验

## Barlow Twins benchmark (Painting91, Pandora, ArtBench, FashionStyle14, Arch) (pretrain_epochs=50, classifier_epochs=100, runs=3) — 2026-07-01 10:30:24

_data_base=`/mnt/codes/data/style/`_

_命令: `./selfsupervised/run_barlowtwins_train_bat.sh` → `barlowtwins_train.py` × 5 数据集_

### Accuracy

| Dataset | num_classes | run1 | run2 | run3 | mean±std | data_root |
|---------|---------|---------|---------|---------|---------|---------|
| Painting91 | 13 | 0.3130 | 0.2962 | 0.3109 | 0.3067±0.0092 | `/mnt/codes/data/style/Painting91` |
| Pandora | 12 | 0.3873 | 0.3893 | 0.3919 | 0.3895±0.0023 | `/mnt/codes/data/style/Pandora` |
| ArtBench | 10 | 0.4957 | 0.4952 | 0.4952 | 0.4954±0.0003 | `/mnt/codes/data/style/Artbench` |
| FashionStyle14 | 14 | 0.3398 | 0.3229 | 0.3654 | 0.3427±0.0214 | `/mnt/codes/data/style/FashionStyle14` |
| Arch | 25 | 0.4446 | 0.4235 | 0.4300 | 0.4327±0.0108 | `/mnt/codes/data/style/Arch` |

### Macro-F1

| Dataset | num_classes | run1 | run2 | run3 | mean±std | data_root |
|---------|---------|---------|---------|---------|---------|---------|
| Painting91 | 13 | 0.2732 | 0.1961 | 0.2390 | 0.2361±0.0386 | `/mnt/codes/data/style/Painting91` |
| Pandora | 12 | 0.3510 | 0.3191 | 0.3398 | 0.3366±0.0162 | `/mnt/codes/data/style/Pandora` |
| ArtBench | 10 | 0.4856 | 0.4853 | 0.4919 | 0.4876±0.0037 | `/mnt/codes/data/style/Artbench` |
| FashionStyle14 | 14 | 0.3145 | 0.2854 | 0.3518 | 0.3172±0.0333 | `/mnt/codes/data/style/FashionStyle14` |
| Arch | 25 | 0.4227 | 0.4028 | 0.4115 | 0.4123±0.0100 | `/mnt/codes/data/style/Arch` |

### Weighted-F1

| Dataset | num_classes | run1 | run2 | run3 | mean±std | data_root |
|---------|---------|---------|---------|---------|---------|---------|
| Painting91 | 13 | 0.2825 | 0.2347 | 0.2602 | 0.2591±0.0239 | `/mnt/codes/data/style/Painting91` |
| Pandora | 12 | 0.3469 | 0.3397 | 0.3393 | 0.3420±0.0043 | `/mnt/codes/data/style/Pandora` |
| ArtBench | 10 | 0.4856 | 0.4853 | 0.4919 | 0.4876±0.0037 | `/mnt/codes/data/style/Artbench` |
| FashionStyle14 | 14 | 0.3121 | 0.2845 | 0.3518 | 0.3161±0.0339 | `/mnt/codes/data/style/FashionStyle14` |
| Arch | 25 | 0.4303 | 0.4080 | 0.4211 | 0.4198±0.0112 | `/mnt/codes/data/style/Arch` |

### Balanced Accuracy

| Dataset | num_classes | run1 | run2 | run3 | mean±std | data_root |
|---------|---------|---------|---------|---------|---------|---------|
| Painting91 | 13 | 0.2922 | 0.2413 | 0.2739 | 0.2691±0.0258 | `/mnt/codes/data/style/Painting91` |
| Pandora | 12 | 0.3854 | 0.3525 | 0.3667 | 0.3682±0.0165 | `/mnt/codes/data/style/Pandora` |
| ArtBench | 10 | 0.4957 | 0.4952 | 0.4952 | 0.4954±0.0003 | `/mnt/codes/data/style/Artbench` |
| FashionStyle14 | 14 | 0.3486 | 0.3245 | 0.3618 | 0.3450±0.0189 | `/mnt/codes/data/style/FashionStyle14` |
| Arch | 25 | 0.4395 | 0.4165 | 0.4202 | 0.4254±0.0124 | `/mnt/codes/data/style/Arch` |

## 汇总总表

| Dataset | num_classes | Accuracy | Macro-F1 | Weighted-F1 | Balanced Accuracy |
|---------|-------------|---------|---------|---------|---------|
| Painting91 | 13 | 0.3067±0.0092 | 0.2361±0.0386 | 0.2591±0.0239 | 0.2691±0.0258 |
| Pandora | 12 | 0.3895±0.0023 | 0.3366±0.0162 | 0.3420±0.0043 | 0.3682±0.0165 |
| ArtBench | 10 | 0.4954±0.0003 | 0.4876±0.0037 | 0.4876±0.0037 | 0.4954±0.0003 |
| FashionStyle14 | 14 | 0.3427±0.0214 | 0.3172±0.0333 | 0.3161±0.0339 | 0.3450±0.0189 |
| Arch | 25 | 0.4327±0.0108 | 0.4123±0.0100 | 0.4198±0.0112 | 0.4254±0.0124 |
