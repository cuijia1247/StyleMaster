# MCCFNet 多数据集多次实验

## MCCFNet benchmark (Painting91, Pandora, ArtBench, FashionStyle14, Arch) (epochs=3, runs=3) — 2026-07-01 10:58:58

_data_base=`/mnt/codes/data/style/`_

_命令: `./MCCFNet/run_mccfnet_train_bat.sh` → `mccfnet_train.py` × 5 数据集_

### Accuracy

| Dataset | num_classes | run1 | run2 | run3 | mean±std | data_root |
|---------|---------|---------|---------|---------|---------|---------|
| Painting91 | 13 | 0.7479 | 0.7479 | 0.7122 | 0.7360±0.0206 | `/mnt/codes/data/style/Painting91` |
| Pandora | 12 | 0.6427 | 0.6303 | 0.6042 | 0.6257±0.0197 | `/mnt/codes/data/style/Pandora` |
| ArtBench | 10 | 0.6222 | 0.6362 | 0.6333 | 0.6306±0.0074 | `/mnt/codes/data/style/Artbench` |
| FashionStyle14 | 14 | 0.7454 | 0.7600 | 0.7653 | 0.7569±0.0103 | `/mnt/codes/data/style/FashionStyle14` |
| Arch | 25 | 0.7090 | 0.7055 | 0.7090 | 0.7078±0.0020 | `/mnt/codes/data/style/Arch` |

### Macro-F1

| Dataset | num_classes | run1 | run2 | run3 | mean±std | data_root |
|---------|---------|---------|---------|---------|---------|---------|
| Painting91 | 13 | 0.7491 | 0.7344 | 0.7137 | 0.7324±0.0178 | `/mnt/codes/data/style/Painting91` |
| Pandora | 12 | 0.6136 | 0.5944 | 0.5938 | 0.6006±0.0113 | `/mnt/codes/data/style/Pandora` |
| ArtBench | 10 | 0.6172 | 0.6353 | 0.6289 | 0.6271±0.0092 | `/mnt/codes/data/style/Artbench` |
| FashionStyle14 | 14 | 0.7439 | 0.7578 | 0.7658 | 0.7558±0.0111 | `/mnt/codes/data/style/FashionStyle14` |
| Arch | 25 | 0.6983 | 0.6901 | 0.6995 | 0.6960±0.0051 | `/mnt/codes/data/style/Arch` |

### Weighted-F1

| Dataset | num_classes | run1 | run2 | run3 | mean±std | data_root |
|---------|---------|---------|---------|---------|---------|---------|
| Painting91 | 13 | 0.7466 | 0.7434 | 0.7132 | 0.7344±0.0184 | `/mnt/codes/data/style/Painting91` |
| Pandora | 12 | 0.6113 | 0.6107 | 0.5845 | 0.6022±0.0153 | `/mnt/codes/data/style/Pandora` |
| ArtBench | 10 | 0.6172 | 0.6353 | 0.6289 | 0.6271±0.0092 | `/mnt/codes/data/style/Artbench` |
| FashionStyle14 | 14 | 0.7422 | 0.7578 | 0.7643 | 0.7548±0.0114 | `/mnt/codes/data/style/FashionStyle14` |
| Arch | 25 | 0.7064 | 0.6997 | 0.7094 | 0.7052±0.0049 | `/mnt/codes/data/style/Arch` |

### Balanced Accuracy

| Dataset | num_classes | run1 | run2 | run3 | mean±std | data_root |
|---------|---------|---------|---------|---------|---------|---------|
| Painting91 | 13 | 0.7414 | 0.7429 | 0.7150 | 0.7331±0.0157 | `/mnt/codes/data/style/Painting91` |
| Pandora | 12 | 0.6333 | 0.6139 | 0.6225 | 0.6232±0.0097 | `/mnt/codes/data/style/Pandora` |
| ArtBench | 10 | 0.6222 | 0.6362 | 0.6333 | 0.6306±0.0074 | `/mnt/codes/data/style/Artbench` |
| FashionStyle14 | 14 | 0.7488 | 0.7621 | 0.7681 | 0.7597±0.0099 | `/mnt/codes/data/style/FashionStyle14` |
| Arch | 25 | 0.6987 | 0.6893 | 0.6965 | 0.6948±0.0049 | `/mnt/codes/data/style/Arch` |

## 汇总总表

| Dataset | num_classes | Accuracy | Macro-F1 | Weighted-F1 | Balanced Accuracy |
|---------|-------------|---------|---------|---------|---------|
| Painting91 | 13 | 0.7360±0.0206 | 0.7324±0.0178 | 0.7344±0.0184 | 0.7331±0.0157 |
| Pandora | 12 | 0.6257±0.0197 | 0.6006±0.0113 | 0.6022±0.0153 | 0.6232±0.0097 |
| ArtBench | 10 | 0.6306±0.0074 | 0.6271±0.0092 | 0.6271±0.0092 | 0.6306±0.0074 |
| FashionStyle14 | 14 | 0.7569±0.0103 | 0.7558±0.0111 | 0.7548±0.0114 | 0.7597±0.0099 |
| Arch | 25 | 0.7078±0.0020 | 0.6960±0.0051 | 0.7052±0.0049 | 0.6948±0.0049 |
