# SimCLR (SSC) 多数据集多次实验

## SimCLR (SSC) benchmark (Painting91, Pandora, ArtBench, FashionStyle14, Arch) (epochs=126, runs=3) — 2026-06-29 20:38:02

_data_base=`/mnt/codes/data/style/`_

_命令: `./selfsupervised/run_simclr_train_bat.sh` → `simclr_train_root.py` × 5 数据集_

### Accuracy

| Dataset | num_classes | run1 | run2 | run3 | mean±std | data_root |
|---------|---------|---------|---------|---------|---------|---------|
| Painting91 | 13 | 0.5546 | 0.5840 | 0.5798 | 0.5728±0.0159 | `/mnt/codes/data/style/Painting91` |
| Pandora | 12 | 0.5069 | 0.4918 | 0.4977 | 0.4988±0.0076 | `/mnt/codes/data/style/Pandora` |
| artbench | 10 | FAILED | FAILED | FAILED | FAILED | `/mnt/codes/data/style/artbench` |
| FashionStyle14 | 14 | 0.4198 | 0.4112 | 0.4022 | 0.4111±0.0088 | `/mnt/codes/data/style/FashionStyle14` |
| Arch | 25 | 0.3843 | 0.4511 | 0.4576 | 0.4310±0.0405 | `/mnt/codes/data/style/Arch` |

### Macro-F1

| Dataset | num_classes | run1 | run2 | run3 | mean±std | data_root |
|---------|---------|---------|---------|---------|---------|---------|
| Painting91 | 13 | 0.5197 | 0.5760 | 0.5676 | 0.5544±0.0304 | `/mnt/codes/data/style/Painting91` |
| Pandora | 12 | 0.4668 | 0.4501 | 0.4656 | 0.4609±0.0093 | `/mnt/codes/data/style/Pandora` |
| artbench | 10 | FAILED | FAILED | FAILED | FAILED | `/mnt/codes/data/style/artbench` |
| FashionStyle14 | 14 | 0.3969 | 0.3963 | 0.3990 | 0.3974±0.0014 | `/mnt/codes/data/style/FashionStyle14` |
| Arch | 25 | 0.3480 | 0.4197 | 0.4305 | 0.3994±0.0449 | `/mnt/codes/data/style/Arch` |

### Weighted-F1

| Dataset | num_classes | run1 | run2 | run3 | mean±std | data_root |
|---------|---------|---------|---------|---------|---------|---------|
| Painting91 | 13 | 0.5396 | 0.5764 | 0.5719 | 0.5626±0.0201 | `/mnt/codes/data/style/Painting91` |
| Pandora | 12 | 0.4602 | 0.4503 | 0.4559 | 0.4555±0.0049 | `/mnt/codes/data/style/Pandora` |
| artbench | 10 | FAILED | FAILED | FAILED | FAILED | `/mnt/codes/data/style/artbench` |
| FashionStyle14 | 14 | 0.3941 | 0.3906 | 0.3960 | 0.3936±0.0027 | `/mnt/codes/data/style/FashionStyle14` |
| Arch | 25 | 0.3491 | 0.4241 | 0.4327 | 0.4020±0.0460 | `/mnt/codes/data/style/Arch` |

### Balanced Accuracy

| Dataset | num_classes | run1 | run2 | run3 | mean±std | data_root |
|---------|---------|---------|---------|---------|---------|---------|
| Painting91 | 13 | 0.5226 | 0.5856 | 0.5698 | 0.5593±0.0328 | `/mnt/codes/data/style/Painting91` |
| Pandora | 12 | 0.4971 | 0.4712 | 0.4873 | 0.4852±0.0131 | `/mnt/codes/data/style/Pandora` |
| artbench | 10 | FAILED | FAILED | FAILED | FAILED | `/mnt/codes/data/style/artbench` |
| FashionStyle14 | 14 | 0.4259 | 0.4216 | 0.4039 | 0.4171±0.0116 | `/mnt/codes/data/style/FashionStyle14` |
| Arch | 25 | 0.3763 | 0.4425 | 0.4484 | 0.4224±0.0400 | `/mnt/codes/data/style/Arch` |

## 汇总总表

| Dataset | num_classes | Accuracy | Macro-F1 | Weighted-F1 | Balanced Accuracy |
|---------|-------------|---------|---------|---------|---------|
| Painting91 | 13 | 0.5728±0.0159 | 0.5544±0.0304 | 0.5626±0.0201 | 0.5593±0.0328 |
| Pandora | 12 | 0.4988±0.0076 | 0.4609±0.0093 | 0.4555±0.0049 | 0.4852±0.0131 |
| artbench | 10 | FAILED | FAILED | FAILED | FAILED |
| FashionStyle14 | 14 | 0.4111±0.0088 | 0.3974±0.0014 | 0.3936±0.0027 | 0.4171±0.0116 |
| Arch | 25 | 0.4310±0.0405 | 0.3994±0.0449 | 0.4020±0.0460 | 0.4224±0.0400 |
