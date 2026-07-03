# ST-SACLF (AdaIN) 多数据集多次实验

## ST-SACLF (AdaIN) benchmark (Painting91, Pandora, ArtBench, FashionStyle14, Arch) (max_iter=10000, clf_epochs=20, runs=3) — 2026-07-01 18:01:51

_data_base=`/mnt/codes/data/style/`_

_命令: `./ST-SACLF-ncc_main/pytorch-AdaIN/run_st_saclf_train_bat.sh` → `train.py` × 5 数据集_

### Accuracy

| Dataset | num_classes | run1 | run2 | run3 | mean±std | data_root |
|---------|---------|---------|---------|---------|---------|---------|
| Painting91 | 13 | 0.6681 | 0.6954 | 0.6828 | 0.6821±0.0137 | `/mnt/codes/data/style/Painting91` |
| Pandora | 12 | 0.5180 | 0.5147 | 0.5180 | 0.5169±0.0019 | `/mnt/codes/data/style/Pandora` |
| ArtBench | 10 | 0.5151 | 0.5145 | 0.5060 | 0.5119±0.0051 | `/mnt/codes/data/style/Artbench` |
| FashionStyle14 | 14 | 0.6234 | 0.6125 | 0.6177 | 0.6178±0.0054 | `/mnt/codes/data/style/FashionStyle14` |
| Arch | 25 | 0.5635 | 0.5670 | 0.5610 | 0.5638±0.0030 | `/mnt/codes/data/style/Arch` |

### Macro-F1

| Dataset | num_classes | run1 | run2 | run3 | mean±std | data_root |
|---------|---------|---------|---------|---------|---------|---------|
| Painting91 | 13 | 0.6829 | 0.7056 | 0.6842 | 0.6909±0.0128 | `/mnt/codes/data/style/Painting91` |
| Pandora | 12 | 0.5204 | 0.5025 | 0.5163 | 0.5130±0.0094 | `/mnt/codes/data/style/Pandora` |
| ArtBench | 10 | 0.5038 | 0.5135 | 0.4956 | 0.5043±0.0090 | `/mnt/codes/data/style/Artbench` |
| FashionStyle14 | 14 | 0.6268 | 0.6118 | 0.6191 | 0.6192±0.0075 | `/mnt/codes/data/style/FashionStyle14` |
| Arch | 25 | 0.5595 | 0.5612 | 0.5601 | 0.5603±0.0009 | `/mnt/codes/data/style/Arch` |

### Weighted-F1

| Dataset | num_classes | run1 | run2 | run3 | mean±std | data_root |
|---------|---------|---------|---------|---------|---------|---------|
| Painting91 | 13 | 0.6687 | 0.6961 | 0.6838 | 0.6829±0.0137 | `/mnt/codes/data/style/Painting91` |
| Pandora | 12 | 0.5166 | 0.5025 | 0.5140 | 0.5110±0.0075 | `/mnt/codes/data/style/Pandora` |
| ArtBench | 10 | 0.5038 | 0.5135 | 0.4956 | 0.5043±0.0090 | `/mnt/codes/data/style/Artbench` |
| FashionStyle14 | 14 | 0.6240 | 0.6082 | 0.6156 | 0.6159±0.0079 | `/mnt/codes/data/style/FashionStyle14` |
| Arch | 25 | 0.5641 | 0.5675 | 0.5635 | 0.5651±0.0022 | `/mnt/codes/data/style/Arch` |

### Balanced Accuracy

| Dataset | num_classes | run1 | run2 | run3 | mean±std | data_root |
|---------|---------|---------|---------|---------|---------|---------|
| Painting91 | 13 | 0.6832 | 0.7024 | 0.6777 | 0.6878±0.0130 | `/mnt/codes/data/style/Painting91` |
| Pandora | 12 | 0.5167 | 0.5053 | 0.5116 | 0.5112±0.0057 | `/mnt/codes/data/style/Pandora` |
| ArtBench | 10 | 0.5151 | 0.5145 | 0.5060 | 0.5119±0.0051 | `/mnt/codes/data/style/Artbench` |
| FashionStyle14 | 14 | 0.6250 | 0.6161 | 0.6225 | 0.6212±0.0046 | `/mnt/codes/data/style/FashionStyle14` |
| Arch | 25 | 0.5585 | 0.5584 | 0.5618 | 0.5596±0.0019 | `/mnt/codes/data/style/Arch` |

## 汇总总表

| Dataset | num_classes | Accuracy | Macro-F1 | Weighted-F1 | Balanced Accuracy |
|---------|-------------|---------|---------|---------|---------|
| Painting91 | 13 | 0.6821±0.0137 | 0.6909±0.0128 | 0.6829±0.0137 | 0.6878±0.0130 |
| Pandora | 12 | 0.5169±0.0019 | 0.5130±0.0094 | 0.5110±0.0075 | 0.5112±0.0057 |
| ArtBench | 10 | 0.5119±0.0051 | 0.5043±0.0090 | 0.5043±0.0090 | 0.5119±0.0051 |
| FashionStyle14 | 14 | 0.6178±0.0054 | 0.6192±0.0075 | 0.6159±0.0079 | 0.6212±0.0046 |
| Arch | 25 | 0.5638±0.0030 | 0.5603±0.0009 | 0.5651±0.0022 | 0.5596±0.0019 |
