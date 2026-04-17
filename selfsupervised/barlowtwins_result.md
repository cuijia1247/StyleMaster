# Barlow Twins 六数据集评测

下列表格由 `python selfsupervised/barlowtwins_train.py --benchmark_all` 自动追加（格式对齐 `simclr_result.md` / `dae_result.md`）。

## Barlow Twins 六数据集 (backbone=resnet50, run=3) — 2026-04-17 22:11:37

_data_base=`/mnt/codes/data/style/`_

_命令: `/home/idtrc/Home_Codes/SubStyleClassfication/selfsupervised/barlowtwins_train.py --benchmark_all --runs 3 --result_md /home/idtrc/Home_Codes/SubStyleClassfication/selfsupervised/barlowtwins_result.md`_

| Dataset | num_classes | run1 | run2 | run3 | mean±std | data_root |
|---------|---------|---------|---------|---------|---------|---------|
| Painting91 | 13 | 0.3130 | 0.3214 | 0.2899 | 0.3081±0.0163 | `/mnt/codes/data/style/Painting91` |
| Pandora | 12 | 0.3932 | 0.3939 | 0.3625 | 0.3832±0.0179 | `/mnt/codes/data/style/Pandora` |
| AVAstyle | 14 | 0.3374 | 0.3239 | 0.3270 | 0.3294±0.0071 | `/mnt/codes/data/style/AVAstyle` |
| FashionStyle14 | 14 | 0.3793 | 0.3706 | 0.3830 | 0.3776±0.0064 | `/mnt/codes/data/style/FashionStyle14` |
| Arch | 25 | 0.3673 | 0.3889 | 0.3954 | 0.3838±0.0147 | `/mnt/codes/data/style/Arch` |
| WebStyle | 10 | 0.3361 | 0.3715 | 0.3682 | 0.3586±0.0195 | `/mnt/codes/data/style/webstyle` |
