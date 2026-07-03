# CLIP-based 多数据集多次实验

## Zero-shot benchmark (Painting91, Pandora, ArtBench, FashionStyle14, Arch) (backbone=ViT-L-14 (local), runs=3) — 2026-07-01 12:24:16

_data_base=`/mnt/codes/data/style/`_

_命令: `./CLIP-based/run_openclip_train_bat.sh` → `openclip_community_variants_train.py` mode=zero_shot × 5 数据集_

### Accuracy

| Dataset | num_classes | run1 | run2 | run3 | mean±std | data_root |
|---------|---------|---------|---------|---------|---------|---------|
| Painting91 | 13 | 0.0546 | 0.0546 | 0.0546 | 0.0546±0.0000 | `/mnt/codes/data/style/Painting91` |
| Pandora | 12 | 0.0523 | 0.0523 | 0.0523 | 0.0523±0.0000 | `/mnt/codes/data/style/Pandora` |
| ArtBench | 10 | 0.0741 | 0.0741 | 0.0741 | 0.0741±0.0000 | `/mnt/codes/data/style/Artbench` |
| FashionStyle14 | 14 | 0.0744 | 0.0744 | 0.0744 | 0.0744±0.0000 | `/mnt/codes/data/style/FashionStyle14` |
| Arch | 25 | 0.0437 | 0.0437 | 0.0437 | 0.0437±0.0000 | `/mnt/codes/data/style/Arch` |

### Macro-F1

| Dataset | num_classes | run1 | run2 | run3 | mean±std | data_root |
|---------|---------|---------|---------|---------|---------|---------|
| Painting91 | 13 | 0.0364 | 0.0364 | 0.0364 | 0.0364±0.0000 | `/mnt/codes/data/style/Painting91` |
| Pandora | 12 | 0.0436 | 0.0436 | 0.0436 | 0.0436±0.0000 | `/mnt/codes/data/style/Pandora` |
| ArtBench | 10 | 0.0465 | 0.0465 | 0.0465 | 0.0465±0.0000 | `/mnt/codes/data/style/Artbench` |
| FashionStyle14 | 14 | 0.0474 | 0.0474 | 0.0474 | 0.0474±0.0000 | `/mnt/codes/data/style/FashionStyle14` |
| Arch | 25 | 0.0245 | 0.0245 | 0.0245 | 0.0245±0.0000 | `/mnt/codes/data/style/Arch` |

### Weighted-F1

| Dataset | num_classes | run1 | run2 | run3 | mean±std | data_root |
|---------|---------|---------|---------|---------|---------|---------|
| Painting91 | 13 | 0.0293 | 0.0293 | 0.0293 | 0.0293±0.0000 | `/mnt/codes/data/style/Painting91` |
| Pandora | 12 | 0.0509 | 0.0509 | 0.0509 | 0.0509±0.0000 | `/mnt/codes/data/style/Pandora` |
| ArtBench | 10 | 0.0465 | 0.0465 | 0.0465 | 0.0465±0.0000 | `/mnt/codes/data/style/Artbench` |
| FashionStyle14 | 14 | 0.0471 | 0.0471 | 0.0471 | 0.0471±0.0000 | `/mnt/codes/data/style/FashionStyle14` |
| Arch | 25 | 0.0254 | 0.0254 | 0.0254 | 0.0254±0.0000 | `/mnt/codes/data/style/Arch` |

### Balanced Accuracy

| Dataset | num_classes | run1 | run2 | run3 | mean±std | data_root |
|---------|---------|---------|---------|---------|---------|---------|
| Painting91 | 13 | 0.0700 | 0.0700 | 0.0700 | 0.0700±0.0000 | `/mnt/codes/data/style/Painting91` |
| Pandora | 12 | 0.0637 | 0.0637 | 0.0637 | 0.0637±0.0000 | `/mnt/codes/data/style/Pandora` |
| ArtBench | 10 | 0.0741 | 0.0741 | 0.0741 | 0.0741±0.0000 | `/mnt/codes/data/style/Artbench` |
| FashionStyle14 | 14 | 0.0734 | 0.0734 | 0.0734 | 0.0734±0.0000 | `/mnt/codes/data/style/FashionStyle14` |
| Arch | 25 | 0.0466 | 0.0466 | 0.0466 | 0.0466±0.0000 | `/mnt/codes/data/style/Arch` |

## 汇总总表

| Dataset | num_classes | Accuracy | Macro-F1 | Weighted-F1 | Balanced Accuracy |
|---------|-------------|---------|---------|---------|---------|
| Painting91 | 13 | 0.0546±0.0000 | 0.0364±0.0000 | 0.0293±0.0000 | 0.0700±0.0000 |
| Pandora | 12 | 0.0523±0.0000 | 0.0436±0.0000 | 0.0509±0.0000 | 0.0637±0.0000 |
| ArtBench | 10 | 0.0741±0.0000 | 0.0465±0.0000 | 0.0465±0.0000 | 0.0741±0.0000 |
| FashionStyle14 | 14 | 0.0744±0.0000 | 0.0474±0.0000 | 0.0471±0.0000 | 0.0734±0.0000 |
| Arch | 25 | 0.0437±0.0000 | 0.0245±0.0000 | 0.0254±0.0000 | 0.0466±0.0000 |

## Linear Probe benchmark (Painting91, Pandora, ArtBench, FashionStyle14, Arch) (backbone=vit_large_patch16_224 (local), classifier_epochs=50, runs=3) — 2026-07-01 12:24:16

_data_base=`/mnt/codes/data/style/`_

_命令: `./CLIP-based/run_openclip_train_bat.sh` → `openclip_community_variants_train.py` mode=linear_probe × 5 数据集_

### Accuracy

| Dataset | num_classes | run1 | run2 | run3 | mean±std | data_root |
|---------|---------|---------|---------|---------|---------|---------|
| Painting91 | 13 | 0.6197 | 0.6155 | 0.6429 | 0.6261±0.0147 | `/mnt/codes/data/style/Painting91` |
| Pandora | 12 | 0.5950 | 0.5911 | 0.5931 | 0.5931±0.0020 | `/mnt/codes/data/style/Pandora` |
| ArtBench | 10 | 0.5897 | 0.5898 | 0.5883 | 0.5893±0.0008 | `/mnt/codes/data/style/Artbench` |
| FashionStyle14 | 14 | 0.7492 | 0.7514 | 0.7473 | 0.7493±0.0021 | `/mnt/codes/data/style/FashionStyle14` |
| Arch | 25 | 0.6814 | 0.6829 | 0.6844 | 0.6829±0.0015 | `/mnt/codes/data/style/Arch` |

### Macro-F1

| Dataset | num_classes | run1 | run2 | run3 | mean±std | data_root |
|---------|---------|---------|---------|---------|---------|---------|
| Painting91 | 13 | 0.6002 | 0.6046 | 0.6333 | 0.6127±0.0179 | `/mnt/codes/data/style/Painting91` |
| Pandora | 12 | 0.5498 | 0.5412 | 0.5462 | 0.5457±0.0043 | `/mnt/codes/data/style/Pandora` |
| ArtBench | 10 | 0.5865 | 0.5871 | 0.5862 | 0.5866±0.0004 | `/mnt/codes/data/style/Artbench` |
| FashionStyle14 | 14 | 0.7518 | 0.7537 | 0.7493 | 0.7516±0.0022 | `/mnt/codes/data/style/FashionStyle14` |
| Arch | 25 | 0.6691 | 0.6711 | 0.6726 | 0.6710±0.0018 | `/mnt/codes/data/style/Arch` |

### Weighted-F1

| Dataset | num_classes | run1 | run2 | run3 | mean±std | data_root |
|---------|---------|---------|---------|---------|---------|---------|
| Painting91 | 13 | 0.6159 | 0.6137 | 0.6404 | 0.6233±0.0148 | `/mnt/codes/data/style/Painting91` |
| Pandora | 12 | 0.5638 | 0.5586 | 0.5582 | 0.5602±0.0031 | `/mnt/codes/data/style/Pandora` |
| ArtBench | 10 | 0.5865 | 0.5871 | 0.5862 | 0.5866±0.0004 | `/mnt/codes/data/style/Artbench` |
| FashionStyle14 | 14 | 0.7487 | 0.7510 | 0.7464 | 0.7487±0.0023 | `/mnt/codes/data/style/FashionStyle14` |
| Arch | 25 | 0.6791 | 0.6803 | 0.6819 | 0.6804±0.0014 | `/mnt/codes/data/style/Arch` |

### Balanced Accuracy

| Dataset | num_classes | run1 | run2 | run3 | mean±std | data_root |
|---------|---------|---------|---------|---------|---------|---------|
| Painting91 | 13 | 0.5939 | 0.5930 | 0.6238 | 0.6036±0.0175 | `/mnt/codes/data/style/Painting91` |
| Pandora | 12 | 0.5753 | 0.5642 | 0.5719 | 0.5705±0.0057 | `/mnt/codes/data/style/Pandora` |
| ArtBench | 10 | 0.5897 | 0.5898 | 0.5883 | 0.5893±0.0008 | `/mnt/codes/data/style/Artbench` |
| FashionStyle14 | 14 | 0.7516 | 0.7538 | 0.7500 | 0.7518±0.0019 | `/mnt/codes/data/style/FashionStyle14` |
| Arch | 25 | 0.6694 | 0.6714 | 0.6738 | 0.6715±0.0022 | `/mnt/codes/data/style/Arch` |

## 汇总总表

| Dataset | num_classes | Accuracy | Macro-F1 | Weighted-F1 | Balanced Accuracy |
|---------|-------------|---------|---------|---------|---------|
| Painting91 | 13 | 0.6261±0.0147 | 0.6127±0.0179 | 0.6233±0.0148 | 0.6036±0.0175 |
| Pandora | 12 | 0.5931±0.0020 | 0.5457±0.0043 | 0.5602±0.0031 | 0.5705±0.0057 |
| ArtBench | 10 | 0.5893±0.0008 | 0.5866±0.0004 | 0.5866±0.0004 | 0.5893±0.0008 |
| FashionStyle14 | 14 | 0.7493±0.0021 | 0.7516±0.0022 | 0.7487±0.0023 | 0.7518±0.0019 |
| Arch | 25 | 0.6829±0.0015 | 0.6710±0.0018 | 0.6804±0.0014 | 0.6715±0.0022 |
