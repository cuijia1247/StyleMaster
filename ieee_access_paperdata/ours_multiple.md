# Ours (SSC-ResNet50) 多数据集多次实验

## Ours (SSC-ResNet50) benchmark (Painting91) (ssc_epochs=100, classifier_iteration=100, runs=3) — 2026-07-03 09:50:44

_data_base=`/mnt/codes/data/style/`_

_命令: `/home/idtrc/Home_Codes/SubStyleClassfication/ieee_ssc_train_resnet.py --benchmark_all --data_base /mnt/codes/data/style --runs 3 --result_md /home/idtrc/Home_Codes/SubStyleClassfication/ieee_access_paperdata/ours_multiple.md --pre_feature_path /home/idtrc/Home_Codes/SubStyleClassfication/pretrainFeatures --model_path /home/idtrc/Home_Codes/SubStyleClassfication/model`_

### Accuracy

| Dataset | num_classes | run1 | run2 | run3 | mean±std | data_root |
|---------|---------|---------|---------|---------|---------|---------|
| Painting91 | 13 | 0.7479 | 0.7605 | 0.7185 | 0.7423±0.0216 | `/mnt/codes/data/style/Painting91` |

### Macro-F1

| Dataset | num_classes | run1 | run2 | run3 | mean±std | data_root |
|---------|---------|---------|---------|---------|---------|---------|
| Painting91 | 13 | 0.7496 | 0.7613 | 0.7253 | 0.7454±0.0184 | `/mnt/codes/data/style/Painting91` |

### Weighted-F1

| Dataset | num_classes | run1 | run2 | run3 | mean±std | data_root |
|---------|---------|---------|---------|---------|---------|---------|
| Painting91 | 13 | 0.7471 | 0.7609 | 0.7196 | 0.7425±0.0210 | `/mnt/codes/data/style/Painting91` |

### Balanced Accuracy

| Dataset | num_classes | run1 | run2 | run3 | mean±std | data_root |
|---------|---------|---------|---------|---------|---------|---------|
| Painting91 | 13 | 0.7526 | 0.7608 | 0.7216 | 0.7450±0.0207 | `/mnt/codes/data/style/Painting91` |

## 汇总总表

| Dataset | num_classes | Accuracy | Macro-F1 | Weighted-F1 | Balanced Accuracy |
|---------|-------------|---------|---------|---------|---------|
| Painting91 | 13 | 0.7423±0.0216 | 0.7454±0.0184 | 0.7425±0.0210 | 0.7450±0.0207 |

## Ours (SSC-ResNet50) benchmark (Pandora) (ssc_epochs=100, classifier_iteration=100, runs=3) — 2026-07-03 19:46:08

_data_base=`/mnt/codes/data/style/`_

_命令: `/home/idtrc/Home_Codes/SubStyleClassfication/ieee_ssc_train_resnet.py --dataset_name Pandora --data_base /mnt/codes/data/style --runs 3 --result_md /home/idtrc/Home_Codes/SubStyleClassfication/ieee_access_paperdata/ours_multiple.md --append_result --pre_feature_path /home/idtrc/Home_Codes/SubStyleClassfication/pretrainFeatures --model_path /home/idtrc/Home_Codes/SubStyleClassfication/model`_

### Accuracy

| Dataset | num_classes | run1 | run2 | run3 | mean±std | data_root |
|---------|---------|---------|---------|---------|---------|---------|
| Pandora | 12 | 0.6179 | 0.6107 | 0.6133 | 0.6140±0.0036 | `/mnt/codes/data/style/Pandora` |

### Macro-F1

| Dataset | num_classes | run1 | run2 | run3 | mean±std | data_root |
|---------|---------|---------|---------|---------|---------|---------|
| Pandora | 12 | 0.5817 | 0.5768 | 0.5778 | 0.5787±0.0026 | `/mnt/codes/data/style/Pandora` |

### Weighted-F1

| Dataset | num_classes | run1 | run2 | run3 | mean±std | data_root |
|---------|---------|---------|---------|---------|---------|---------|
| Pandora | 12 | 0.5931 | 0.5823 | 0.5871 | 0.5875±0.0054 | `/mnt/codes/data/style/Pandora` |

### Balanced Accuracy

| Dataset | num_classes | run1 | run2 | run3 | mean±std | data_root |
|---------|---------|---------|---------|---------|---------|---------|
| Pandora | 12 | 0.5933 | 0.5907 | 0.5849 | 0.5896±0.0043 | `/mnt/codes/data/style/Pandora` |

## 汇总总表

| Dataset | num_classes | Accuracy | Macro-F1 | Weighted-F1 | Balanced Accuracy |
|---------|-------------|---------|---------|---------|---------|
| Pandora | 12 | 0.6140±0.0036 | 0.5787±0.0026 | 0.5875±0.0054 | 0.5896±0.0043 |

## Ours (SSC-ResNet50) benchmark (ArtBench) (ssc_epochs=100, classifier_iteration=100, runs=3) — 2026-07-03 19:46:10

_data_base=`/mnt/codes/data/style/`_

_命令: `/home/idtrc/Home_Codes/SubStyleClassfication/ieee_ssc_train_resnet.py --dataset_name Artbench --data_base /mnt/codes/data/style --runs 3 --result_md /home/idtrc/Home_Codes/SubStyleClassfication/ieee_access_paperdata/ours_multiple.md --append_result --pre_feature_path /home/idtrc/Home_Codes/SubStyleClassfication/pretrainFeatures --model_path /home/idtrc/Home_Codes/SubStyleClassfication/model`_

### Accuracy

| Dataset | num_classes | run1 | run2 | run3 | mean±std | data_root |
|---------|---------|---------|---------|---------|---------|---------|
| ArtBench | 10 | FAILED | FAILED | FAILED | FAILED | `/mnt/codes/data/style/Artbench` |

### Macro-F1

| Dataset | num_classes | run1 | run2 | run3 | mean±std | data_root |
|---------|---------|---------|---------|---------|---------|---------|
| ArtBench | 10 | FAILED | FAILED | FAILED | FAILED | `/mnt/codes/data/style/Artbench` |

### Weighted-F1

| Dataset | num_classes | run1 | run2 | run3 | mean±std | data_root |
|---------|---------|---------|---------|---------|---------|---------|
| ArtBench | 10 | FAILED | FAILED | FAILED | FAILED | `/mnt/codes/data/style/Artbench` |

### Balanced Accuracy

| Dataset | num_classes | run1 | run2 | run3 | mean±std | data_root |
|---------|---------|---------|---------|---------|---------|---------|
| ArtBench | 10 | FAILED | FAILED | FAILED | FAILED | `/mnt/codes/data/style/Artbench` |

## 汇总总表

| Dataset | num_classes | Accuracy | Macro-F1 | Weighted-F1 | Balanced Accuracy |
|---------|-------------|---------|---------|---------|---------|
| ArtBench | 10 | FAILED | FAILED | FAILED | FAILED |

## Ours (SSC-ResNet50) benchmark (FashionStyle14) (ssc_epochs=100, classifier_iteration=100, runs=3) — 2026-07-04 01:00:37

_data_base=`/mnt/codes/data/style/`_

_命令: `/home/idtrc/Home_Codes/SubStyleClassfication/ieee_ssc_train_resnet.py --dataset_name FashionStyle14 --data_base /mnt/codes/data/style --runs 3 --result_md /home/idtrc/Home_Codes/SubStyleClassfication/ieee_access_paperdata/ours_multiple.md --append_result --pre_feature_path /home/idtrc/Home_Codes/SubStyleClassfication/pretrainFeatures --model_path /home/idtrc/Home_Codes/SubStyleClassfication/model`_

### Accuracy

| Dataset | num_classes | run1 | run2 | run3 | mean±std | data_root |
|---------|---------|---------|---------|---------|---------|---------|
| FashionStyle14 | 14 | 0.7003 | 0.7015 | 0.6996 | 0.7005±0.0009 | `/mnt/codes/data/style/FashionStyle14` |

### Macro-F1

| Dataset | num_classes | run1 | run2 | run3 | mean±std | data_root |
|---------|---------|---------|---------|---------|---------|---------|
| FashionStyle14 | 14 | 0.7026 | 0.7039 | 0.7004 | 0.7023±0.0018 | `/mnt/codes/data/style/FashionStyle14` |

### Weighted-F1

| Dataset | num_classes | run1 | run2 | run3 | mean±std | data_root |
|---------|---------|---------|---------|---------|---------|---------|
| FashionStyle14 | 14 | 0.7002 | 0.7019 | 0.6983 | 0.7002±0.0018 | `/mnt/codes/data/style/FashionStyle14` |

### Balanced Accuracy

| Dataset | num_classes | run1 | run2 | run3 | mean±std | data_root |
|---------|---------|---------|---------|---------|---------|---------|
| FashionStyle14 | 14 | 0.7024 | 0.7034 | 0.7003 | 0.7020±0.0016 | `/mnt/codes/data/style/FashionStyle14` |

## 汇总总表

| Dataset | num_classes | Accuracy | Macro-F1 | Weighted-F1 | Balanced Accuracy |
|---------|-------------|---------|---------|---------|---------|
| FashionStyle14 | 14 | 0.7005±0.0009 | 0.7023±0.0018 | 0.7002±0.0018 | 0.7020±0.0016 |

## Ours (SSC-ResNet50) benchmark (Arch) (ssc_epochs=100, classifier_iteration=100, runs=3) — 2026-07-04 05:08:27

_data_base=`/mnt/codes/data/style/`_

_命令: `/home/idtrc/Home_Codes/SubStyleClassfication/ieee_ssc_train_resnet.py --dataset_name Arch --data_base /mnt/codes/data/style --runs 3 --result_md /home/idtrc/Home_Codes/SubStyleClassfication/ieee_access_paperdata/ours_multiple.md --append_result --pre_feature_path /home/idtrc/Home_Codes/SubStyleClassfication/pretrainFeatures --model_path /home/idtrc/Home_Codes/SubStyleClassfication/model`_

### Accuracy

| Dataset | num_classes | run1 | run2 | run3 | mean±std | data_root |
|---------|---------|---------|---------|---------|---------|---------|
| Arch | 25 | 0.6794 | 0.6809 | 0.6774 | 0.6792±0.0018 | `/mnt/codes/data/style/Arch` |

### Macro-F1

| Dataset | num_classes | run1 | run2 | run3 | mean±std | data_root |
|---------|---------|---------|---------|---------|---------|---------|
| Arch | 25 | 0.6704 | 0.6695 | 0.6688 | 0.6696±0.0008 | `/mnt/codes/data/style/Arch` |

### Weighted-F1

| Dataset | num_classes | run1 | run2 | run3 | mean±std | data_root |
|---------|---------|---------|---------|---------|---------|---------|
| Arch | 25 | 0.6779 | 0.6760 | 0.6748 | 0.6762±0.0016 | `/mnt/codes/data/style/Arch` |

### Balanced Accuracy

| Dataset | num_classes | run1 | run2 | run3 | mean±std | data_root |
|---------|---------|---------|---------|---------|---------|---------|
| Arch | 25 | 0.6717 | 0.6710 | 0.6717 | 0.6715±0.0004 | `/mnt/codes/data/style/Arch` |

## 汇总总表

| Dataset | num_classes | Accuracy | Macro-F1 | Weighted-F1 | Balanced Accuracy |
|---------|-------------|---------|---------|---------|---------|
| Arch | 25 | 0.6792±0.0018 | 0.6696±0.0008 | 0.6762±0.0016 | 0.6715±0.0004 |
