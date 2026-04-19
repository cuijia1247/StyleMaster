
---
## 批次: 2026-04-18 10:45:15

_ssc_train_densenet169.parameter_load(): epochs=20, batch=128, offset_bs=512, base_lr=0.001, image_size=224, classifier_iteration=200, classifier_lr=4e-05, classifier_train_gap=2, classifier_test_gap=2, ssc_io=1664/1664, backbone_cache_workers=8, dataloader_num_workers=8, classifier_cache_k=8; SSCtrain outer_iterations=2, batch_runs=5, 早停=禁用_

### 各次明细

| 数据集 | Run | 最佳准确率 | 训练时长(min) | 状态 |
|--------|-----|-----------|--------------|------|
| Painting91 | 1 | 0.7752 | 13.6 | OK |
| Painting91 | 2 | 0.7794 | 13.4 | OK |
| Painting91 | 3 | 0.7815 | 13.4 | OK |
| Painting91 | 4 | 0.7668 | 13.6 | OK |
| Painting91 | 5 | 0.7731 | 13.4 | OK |

**Painting91** 汇总: `0.7752 ± 0.0058` (runs=5, total=67.2min, acc_list=['0.7752', '0.7794', '0.7815', '0.7668', '0.7731'])

| Pandora | 1 | 0.5905 | 34.6 | OK |
| Pandora | 2 | 0.5937 | 35.0 | OK |
| Pandora | 3 | 0.5885 | 35.1 | OK |
| Pandora | 4 | 0.5905 | 34.9 | OK |
| Pandora | 5 | 0.5924 | 35.0 | OK |

**Pandora** 汇总: `0.5911 ± 0.0020` (runs=5, total=174.6min, acc_list=['0.5905', '0.5937', '0.5885', '0.5905', '0.5924'])

| AVAstyle | 1 | 0.5265 | 48.6 | OK |
| AVAstyle | 2 | 0.5305 | 49.2 | OK |
| AVAstyle | 3 | 0.5364 | 49.7 | OK |
| AVAstyle | 4 | 0.5305 | 49.3 | OK |
| AVAstyle | 5 | 0.5346 | 50.2 | OK |

**AVAstyle** 汇总: `0.5317 ± 0.0039` (runs=5, total=247.0min, acc_list=['0.5265', '0.5305', '0.5364', '0.5305', '0.5346'])

| Arch | 1 | 0.6678 | 47.3 | OK |
| Arch | 2 | 0.6688 | 47.4 | OK |
| Arch | 3 | 0.6779 | 48.1 | OK |
| Arch | 4 | 0.6698 | 47.4 | OK |
| Arch | 5 | 0.6638 | 47.6 | OK |

**Arch** 汇总: `0.6696 ± 0.0051` (runs=5, total=237.7min, acc_list=['0.6678', '0.6688', '0.6779', '0.6698', '0.6638'])

