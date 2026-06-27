# VICReg 多数据集多次实验

## VICReg benchmark (Painting91, Pandora, ArtBench, FashionStyle14, Arch) (epochs=120, runs=3) — 2026-06-26 08:08:41

_data_base=`/mnt/codes/data/style/`_

_命令: `./selfsupervised/run_vicreg_train_bat.sh`_

_日志: `selfsupervised/logs/vicreg_bat_20260625_222949.log`（FashionStyle14 仅完成 run1，run2 中断；Arch 未开始；ArtBench 因 SscDataset 目录结构失败）_

### Accuracy

| Dataset | num_classes | run1 | run2 | run3 | mean±std | data_root |
|---------|---------|---------|---------|---------|---------|---------|
| Painting91 | 13 | 0.6387 | 0.6513 | 0.6618 | 0.6506±0.0116 | `/mnt/codes/data/style/Painting91` |
| Pandora | 12 | 0.3919 | 0.4004 | 0.4344 | 0.4089±0.0225 | `/mnt/codes/data/style/Pandora` |
| ArtBench | 10 | FAILED | FAILED | FAILED | FAILED | `/mnt/codes/data/style/artbench-10-imagefolder-split` |
| FashionStyle14 | 14 | 0.1675 | FAILED | FAILED | 0.1675±0.0000 | `/mnt/codes/data/style/FashionStyle14` |
| Arch | 25 | FAILED | FAILED | FAILED | FAILED | `/mnt/codes/data/style/Arch` |

### Macro-F1

| Dataset | num_classes | run1 | run2 | run3 | mean±std | data_root |
|---------|---------|---------|---------|---------|---------|---------|
| Painting91 | 13 | 0.6325 | 0.6482 | 0.6567 | 0.6458±0.0123 | `/mnt/codes/data/style/Painting91` |
| Pandora | 12 | 0.2857 | 0.2748 | 0.3226 | 0.2944±0.0251 | `/mnt/codes/data/style/Pandora` |
| ArtBench | 10 | FAILED | FAILED | FAILED | FAILED | `/mnt/codes/data/style/artbench-10-imagefolder-split` |
| FashionStyle14 | 14 | 0.1146 | FAILED | FAILED | 0.1146±0.0000 | `/mnt/codes/data/style/FashionStyle14` |
| Arch | 25 | FAILED | FAILED | FAILED | FAILED | `/mnt/codes/data/style/Arch` |

### Weighted-F1

| Dataset | num_classes | run1 | run2 | run3 | mean±std | data_root |
|---------|---------|---------|---------|---------|---------|---------|
| Painting91 | 13 | 0.6346 | 0.6508 | 0.6611 | 0.6488±0.0134 | `/mnt/codes/data/style/Painting91` |
| Pandora | 12 | 0.3018 | 0.2974 | 0.3475 | 0.3156±0.0277 | `/mnt/codes/data/style/Pandora` |
| ArtBench | 10 | FAILED | FAILED | FAILED | FAILED | `/mnt/codes/data/style/artbench-10-imagefolder-split` |
| FashionStyle14 | 14 | 0.1136 | FAILED | FAILED | 0.1136±0.0000 | `/mnt/codes/data/style/FashionStyle14` |
| Arch | 25 | FAILED | FAILED | FAILED | FAILED | `/mnt/codes/data/style/Arch` |

### Balanced Accuracy

| Dataset | num_classes | run1 | run2 | run3 | mean±std | data_root |
|---------|---------|---------|---------|---------|---------|---------|
| Painting91 | 13 | 0.6389 | 0.6389 | 0.6484 | 0.6421±0.0055 | `/mnt/codes/data/style/Painting91` |
| Pandora | 12 | 0.3551 | 0.3576 | 0.3862 | 0.3663±0.0173 | `/mnt/codes/data/style/Pandora` |
| ArtBench | 10 | FAILED | FAILED | FAILED | FAILED | `/mnt/codes/data/style/artbench-10-imagefolder-split` |
| FashionStyle14 | 14 | 0.1687 | FAILED | FAILED | 0.1687±0.0000 | `/mnt/codes/data/style/FashionStyle14` |
| Arch | 25 | FAILED | FAILED | FAILED | FAILED | `/mnt/codes/data/style/Arch` |

## 汇总总表

| Dataset | num_classes | Accuracy | Macro-F1 | Weighted-F1 | Balanced Accuracy |
|---------|-------------|---------|---------|---------|---------|
| Painting91 | 13 | 0.6506±0.0116 | 0.6458±0.0123 | 0.6488±0.0134 | 0.6421±0.0055 |
| Pandora | 12 | 0.4089±0.0225 | 0.2944±0.0251 | 0.3156±0.0277 | 0.3663±0.0173 |
| ArtBench | 10 | FAILED | FAILED | FAILED | FAILED |
| FashionStyle14 | 14 | 0.1675±0.0000 | 0.1146±0.0000 | 0.1136±0.0000 | 0.1687±0.0000 |
| Arch | 25 | FAILED | FAILED | FAILED | FAILED |
