# Bootstrap 对比实验 (Painting91 test)

## run — 2026-07-16 22:09:35

_test_dir: `/mnt/codes/data/style/Painting91/test`_
_ST-SACLF decoder: `/home/idtrc/Home_Codes/SubStyleClassfication/ST-SACLF-ncc_main/experiments/adain_decoders/decoder_iter_10000_Painting91_run1.pth`_
_Ours base: `/home/idtrc/Home_Codes/SubStyleClassfication/model/ssc-Painting91-SSC-resnet50-2026-07-03-08-54-58-run0-iteration-0-accuracy-7101-SSC-base-best.pth`_
_Ours classifier: `/home/idtrc/Home_Codes/SubStyleClassfication/model/ssc-Painting91-SSC-resnet50-2026-07-03-08-54-58-run0-iteration-0-accuracy-7101-SSC-classifier-best.pth`_
_bootstrap: n=10000, seed=42_

### Accuracy 点估计

| Model | AC | N |
|-------|-----|---|
| ST-SACLF (AdaIN) | 0.6870 | 476 |
| Ours (SSC-ResNet50) | 0.7038 | 476 |

### Bootstrap 95% CI（Accuracy）

| Model | AC | 95% CI |
|-------|-----|--------|
| ST-SACLF (AdaIN) | 0.6870 | [0.6450, 0.7269] |
| Ours (SSC-ResNet50) | 0.7038 | [0.6618, 0.7437] |

### 配对 Bootstrap（Ours − ST-SACLF, Accuracy）

- Δ mean: +0.0169
- 95% CI: [-0.0252, +0.0588]
- p-value (two-sided): 0.4648
