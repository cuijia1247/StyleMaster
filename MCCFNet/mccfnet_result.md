# MCCFNet 六数据集 benchmark 结果

本文件由 `MCCFNet/mccfnet_train.py --benchmark_all`（默认 `--benchmark_runs 5`，与 `run_mccfnet_train_bat.sh` 一致）**追加**写入表格。

每个数据集独立训练 **5 遍**，列 `run1`…`run5` 为各次测试集最佳准确率；`mean±std` 为这 5 次的样本均值与样本标准差（`ddof=1`）；若某次失败则记为 `FAILED`，汇总时用 `nanmean` / `nanstd`。

_data_base 与各数据集路径以每次运行追加段为准。_

## MCCFNet 六数据集 (DenseNet169+RWP, epochs=3, 每库 5 次) — 2026-04-18 23:55:07

_data_base=`/mnt/codes/data/style/`_

_命令: `/home/idtrc/Home_Codes/SubStyleClassfication/MCCFNet/mccfnet_train.py --benchmark_all --benchmark_runs 5 --epochs 3 --data_base /mnt/codes/data/style/ --result_md /home/idtrc/Home_Codes/SubStyleClassfication/MCCFNet/mccfnet_result.md`_

| Dataset | num_classes | run1 | run2 | run3 | run4 | run5 | mean±std | data_root |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Painting91 | 13 | 0.7395 | 0.7437 | 0.7710 | 0.7752 | 0.7500 | 0.7559±0.0162 | `/mnt/codes/data/style/Painting91` |
| Pandora | 12 | 0.6153 | 0.6153 | 0.6316 | 0.6290 | 0.6055 | 0.6193±0.0108 | `/mnt/codes/data/style/Pandora` |
| AVAstyle | 14 | 0.5355 | 0.5310 | 0.5225 | 0.5283 | 0.5270 | 0.5288±0.0048 | `/mnt/codes/data/style/AVAstyle` |
| FashionStyle14 | 14 | 0.7781 | 0.7627 | 0.7807 | 0.7664 | 0.7739 | 0.7724±0.0076 | `/mnt/codes/data/style/FashionStyle14` |
| Arch | 25 | 0.7130 | 0.6969 | 0.7030 | 0.7130 | 0.7100 | 0.7072±0.0070 | `/mnt/codes/data/style/Arch` |
| WebStyle | 10 | 0.6933 | 0.6773 | 0.7110 | 0.6874 | 0.6933 | 0.6925±0.0123 | `/mnt/codes/data/style/webstyle` |
