# SimCLR 六数据集评测

下列表格由 `python selfsupervised/simclr_train.py --benchmark_all` 自动追加（格式对齐 `denoise/dae_result.md`）。

## SimCLR 六数据集 (backbone=resnet50, run=3) — 2026-04-17 09:12:10

_data_base=`/mnt/codes/data/style/`_

_命令: `/home/idtrc/Home_Codes/SubStyleClassfication/selfsupervised/simclr_train.py --benchmark_all --runs 3 --result_md /home/idtrc/Home_Codes/SubStyleClassfication/selfsupervised/simclr_result.md`_

| Dataset | num_classes | run1 | run2 | run3 | mean±std | data_root |
|---------|---------|---------|---------|---------|---------|---------|
| Painting91 | 13 | 0.4559 | 0.4517 | 0.4475 | 0.4517±0.0042 | `/mnt/codes/data/style/Painting91` |
| Pandora | 12 | 0.4742 | 0.4820 | 0.4631 | 0.4731±0.0095 | `/mnt/codes/data/style/Pandora` |
| AVAstyle | 14 | 0.3576 | 0.3877 | 0.3522 | 0.3658±0.0191 | `/mnt/codes/data/style/AVAstyle` |
| FashionStyle14 | 14 | 0.3725 | 0.3785 | 0.3872 | 0.3794±0.0074 | `/mnt/codes/data/style/FashionStyle14` |
| Arch | 25 | 0.4611 | 0.4686 | 0.4727 | 0.4675±0.0059 | `/mnt/codes/data/style/Arch` |
| WebStyle | 10 | 0.4928 | 0.5291 | 0.5063 | 0.5094±0.0183 | `/mnt/codes/data/style/webstyle` |
