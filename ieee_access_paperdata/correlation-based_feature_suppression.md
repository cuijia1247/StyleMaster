# Correlation-based Feature Suppression 实验

## Painting91 benchmark — mean±std over 3 runs (classifier_iteration=10, classifier_lr=3e-05) — 2026-07-16 11:22:49

_SSC-base: `ieee_access_paperdata/models/ssc-Painting91-SSC-resnet50-2026-07-03-09-32-06-run2-iteration-0-accuracy-7353-SSC-base-best.pth`_
_data_root: `/mnt/codes/data/style/Painting91/`_

_命令: `ieee_access_codes/correlation-based_feature_suppression.py --classifier_iteration 10 --repeat_runs 3 --datasets Painting91 FashionStyle14 Arch --model_path ieee_access_paperdata/models/new_models_for_ieee --result_md ieee_access_paperdata/correlation-based_feature_suppression.md`_

| Classifier | Accuracy (mean±std) | Macro-F1 (mean±std) | Weighted-F1 (mean±std) | Balanced Acc (mean±std) |
|------------|---------------------|---------------------|------------------------|-------------------------|
| EfficientClassifier | 0.6625±0.0036 | 0.6467±0.0070 | 0.6598±0.0045 | 0.6408±0.0067 |
| NoSuppressClassifier | 0.6730±0.0026 | 0.6522±0.0092 | 0.6674±0.0039 | 0.6469±0.0058 |
| RandomSuppressClassifier | 0.6646±0.0120 | 0.6434±0.0188 | 0.6590±0.0152 | 0.6399±0.0153 |
| LowCorClassifier | 0.6681±0.0112 | 0.6587±0.0159 | 0.6677±0.0128 | 0.6524±0.0123 |
| HighCorClassifier | 0.6506±0.0069 | 0.6387±0.0099 | 0.6470±0.0075 | 0.6366±0.0097 |

<details><summary>各 run 明细 (Accuracy)</summary>

| Classifier | Run 1 | Run 2 | Run 3 |
|------------|-------|-------|-------|
| EfficientClassifier | 0.6639 | 0.6576 | 0.6660 |
| NoSuppressClassifier | 0.6765 | 0.6702 | 0.6723 |
| RandomSuppressClassifier | 0.6492 | 0.6786 | 0.6660 |
| LowCorClassifier | 0.6555 | 0.6828 | 0.6660 |
| HighCorClassifier | 0.6555 | 0.6408 | 0.6555 |

</details>

## FashionStyle14 benchmark — mean±std over 3 runs (classifier_iteration=10, classifier_lr=3e-05) — 2026-07-16 11:28:23

_SSC-base: `ieee_access_paperdata/models/ssc-FashionStyle14-SSC-resnet50-2026-07-03-23-15-33-run2-iteration-0-accuracy-7015-SSC-base-best.pth`_
_data_root: `/mnt/codes/data/style/FashionStyle14/`_

| Classifier | Accuracy (mean±std) | Macro-F1 (mean±std) | Weighted-F1 (mean±std) | Balanced Acc (mean±std) |
|------------|---------------------|---------------------|------------------------|-------------------------|
| EfficientClassifier | 0.6622±0.0041 | 0.6638±0.0064 | 0.6617±0.0070 | 0.6642±0.0049 |
| NoSuppressClassifier | 0.6657±0.0017 | 0.6639±0.0031 | 0.6615±0.0038 | 0.6694±0.0039 |
| RandomSuppressClassifier | 0.6604±0.0020 | 0.6648±0.0032 | 0.6632±0.0032 | 0.6607±0.0015 |
| LowCorClassifier | 0.6685±0.0055 | 0.6645±0.0053 | 0.6613±0.0053 | 0.6749±0.0052 |
| HighCorClassifier | 0.6669±0.0069 | 0.6694±0.0071 | 0.6680±0.0070 | 0.6674±0.0069 |

<details><summary>各 run 明细 (Accuracy)</summary>

| Classifier | Run 1 | Run 2 | Run 3 |
|------------|-------|-------|-------|
| EfficientClassifier | 0.6677 | 0.6609 | 0.6579 |
| NoSuppressClassifier | 0.6639 | 0.6680 | 0.6650 |
| RandomSuppressClassifier | 0.6605 | 0.6628 | 0.6579 |
| LowCorClassifier | 0.6733 | 0.6714 | 0.6609 |
| HighCorClassifier | 0.6767 | 0.6620 | 0.6620 |

</details>

## Arch benchmark — mean±std over 3 runs (classifier_iteration=10, classifier_lr=3e-05) — 2026-07-16 11:31:55

_SSC-base: `ieee_access_paperdata/models/ssc-Arch-SSC-resnet50-2026-07-04-03-45-21-run2-iteration-0-accuracy-6829-SSC-base-best.pth`_
_data_root: `/mnt/codes/data/style/Arch/`_

| Classifier | Accuracy (mean±std) | Macro-F1 (mean±std) | Weighted-F1 (mean±std) | Balanced Acc (mean±std) |
|------------|---------------------|---------------------|------------------------|-------------------------|
| EfficientClassifier | 0.6312±0.0033 | 0.6096±0.0042 | 0.6205±0.0039 | 0.6178±0.0048 |
| NoSuppressClassifier | 0.6254±0.0033 | 0.6055±0.0022 | 0.6159±0.0026 | 0.6116±0.0028 |
| RandomSuppressClassifier | 0.6269±0.0030 | 0.6052±0.0011 | 0.6153±0.0027 | 0.6125±0.0018 |
| LowCorClassifier | 0.6327±0.0071 | 0.6132±0.0073 | 0.6232±0.0074 | 0.6202±0.0073 |
| HighCorClassifier | 0.6310±0.0027 | 0.6100±0.0023 | 0.6198±0.0025 | 0.6192±0.0035 |

<details><summary>各 run 明细 (Accuracy)</summary>

| Classifier | Run 1 | Run 2 | Run 3 |
|------------|-------|-------|-------|
| EfficientClassifier | 0.6342 | 0.6267 | 0.6327 |
| NoSuppressClassifier | 0.6217 | 0.6297 | 0.6247 |
| RandomSuppressClassifier | 0.6227 | 0.6287 | 0.6292 |
| LowCorClassifier | 0.6382 | 0.6227 | 0.6372 |
| HighCorClassifier | 0.6342 | 0.6277 | 0.6312 |

</details>
