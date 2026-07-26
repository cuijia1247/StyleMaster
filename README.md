# SSC — Sub-Style Classification

**Author:** cuijia1247 | **Started:** Oct. 2024 | **Current Version:** Jul. 2026（最后更新 2026-07-26）

> Paper coming soon.  
> **GitHub 里程碑标签：** `ieee_code_done`（IEEE Access 实验代码与论文数据汇总已完成）

---

## 项目简介

StyleMaster 是一个面向风格的特征学习框架，由两部分组成：
- **Style Consensus Learning (SCL)**：主风格特征学习
- **Sub-Style Classification (SSC)**：子风格细粒度分类

SSC 核心思想：将同一幅画的两个随机裁剪子图（view1 / view2）送入 SSC 编码器，利用自监督损失约束特征空间，再训练轻量分类头完成风格判别。

支持数据集：`Painting91` · `Pandora` · `ArtBench`（目录名 `Artbench`）· `FashionStyle14` · `Arch` · `AVAstyle` · `WikiArt3` · `WebStyle`（`webstyle`）等。

论文复现常用 **五数据集 benchmark**：Painting91、Pandora、ArtBench、FashionStyle14、Arch；数据根默认 `/mnt/codes/data/style/`，每库 **runs=3**，报告 **Accuracy / Macro-F1 / Weighted-F1 / Balanced Accuracy** 四项指标，汇总见 `ieee_access_paperdata/`。

---

## 目录结构

```
SubStyleClassfication/
├── ssc/                              # SSC 核心模块
│   ├── Sscreg.py                     # ResNet-based SscReg 模型
│   ├── Sscreg_transformer.py         # Transformer-based SscReg（Swin / ViT）
│   ├── Backend.py                    # ResNet backbone 封装
│   ├── classifier.py                 # 分类头（Classifier / EfficientClassifier）
│   ├── classifier_enhance.py         # 增强版分类头（StyleEnhancer 门控 + 多路融合）
│   ├── classifier_enhance_add.py     # add 系列四路融合 + SingleViewStyleEnhancer；EfficientRWPClassifier（head 内 RWP）
│   ├── classifier_ieee.py            # IEEE 论文消融：全通道 / 无抑制 / 随机 / 低相关 / 高相关通道抑制分类头
│   ├── Sscreg_densenet169.py         # DenseNet169 冻结骨干 + 6ch 投影 SSC 编码器（1664→1664）
│   ├── classifier_original.py        # 原始分类头存档
│   ├── utils.py                      # 原版损失函数（VICReg + 正交化）
│   └── utils_add.py                  # add 版损失函数（BarlowTwins + SupCon）
├── ieee_access_codes/                # IEEE Access 论文专用脚本（标签 ieee_code_done）
│   ├── ssc_predict_ieee.py           # SSC-ResNet50 测试集推理、推理计时与 failure case 收集
│   ├── ieee_bootstrap.py             # Bootstrap 显著性检验
│   ├── correlation-based_feature_suppression.py  # 通道相关抑制消融实验
│   ├── plot_ieee_ssc_confusion_matrix.py        # 混淆矩阵可视化
│   ├── run_ieee_ssc_train_bat.sh     # 五库 SSC benchmark 后台批量
│   └── manage_ieee_ssc_train_bat.sh
├── utils/                            # 工具脚本
│   ├── pretrainFeatureExtraction.py  # 预训练特征提取与加载
│   ├── image_processing.py           # 图像处理工具
│   ├── styleLevelCal.py              # 风格层级计算
│   └── trainTestSplit.py             # 数据集划分
├── barlowtwins/                      # Barlow Twins 对比实现
├── simclr/                           # SimCLR 对比实现
├── byol/                             # BYOL 对比实现
├── simsiam/                          # SimSiam 对比实现
├── I-JEPA-main/                      # I-JEPA 对比实现
├── HR/                               # 数据清洗、阈值分析脚本
├── denoise/                          # 风格共识 / 去噪自编码 / ConCURL 对比实验（冻结主干特征 + 轻量头）
│   ├── SSCAE.py / sscae_train.py     # CSCAE（K 路 SCAE 共识）+ 训练与六数据集评测
│   ├── DAE.py / dae_train.py         # 堆叠 DAE（SDAE）+ 训练与六数据集评测
│   ├── ConCURL.py / concurl_train.py # 投影 MLP + 分类头（ConCURL 式）+ 训练与六数据集评测
│   └── *_result.md                   # 批量评测结果（本地生成，默认不提交）
├── selfsupervised/                   # SimCLR / Barlow Twins 等自监督基线（五数据集 benchmark）
│   ├── barlowtwins_train.py          # Barlow Twins 预训练 + 线性探针
│   ├── run_*_train_bat.sh            # nohup 批量训练（SSH 断连可续跑）
│   ├── manage_*_train_bat.sh         # 启停 / tail 日志 / 查看结果表
│   └── logs/                         # 批量运行日志（本地，默认不提交）
├── CLIP-based/                       # OpenCLIP 冻结编码器 + 线性探针 / 零样本
│   ├── openclip_community_variants_train.py
│   └── run_openclip_train_bat.sh / manage_openclip_train_bat.sh
├── ST-SACLF-ncc_main/                # ST-SACLF / AdaIN 风格迁移与分类评估
│   ├── models/                       # vgg_normalised.pth 等本地权重（不提交 git）
│   ├── experiments/                  # AdaIN decoder 训练输出（不提交 git）
│   └── pytorch-AdaIN/
│       ├── train.py                  # AdaIN 训练 + 冻结 VGG 线性评估
│       └── run_st_saclf_train_bat.sh / manage_st_saclf_train_bat.sh
├── ieee_access_paperdata/            # 论文实验汇总 Markdown / 图表 / 错误案例分析
│   ├── ours_multiple.md              # Ours（SSC-ResNet50）五库 benchmark
│   ├── simclr_multiple.md / BarlowTwins_multiple.md / MCCFNet_multiple.md
│   ├── clip-based_multiple.md / ST-SACLF_multiple.md / vit_l_16_multiple.md
│   ├── ssc_failure_case_list.md      # SSC 测试集预测错误样本
│   ├── ivt_failure_case_list.md      # ViT-L/16 传统探针测试集错误样本
│   ├── qualitative_vis_list.md       # ViT 错 / SSC 对 — 定性可视化候选清单
│   ├── ieee_bootstrap.md             # Bootstrap 检验结果
│   ├── correlation-based_feature_suppression.md  # 通道抑制消融结果
│   └── ours_ssc_confusion_matrix_*.png         # 各数据集混淆矩阵图
├── simclr_train_root.py              # SimCLR（ResNet50）预训练 + 线性探针 + 五库 benchmark
├── metaclip_train.py                 # MetaCLIP 冻结特征 + 线性分类头
├── traditional_train.py              # 传统 backbone 线性探针；多数据集 failure case + 推理计时
├── ssc_train_densenet169.py          # SSC + 冻结 DenseNet169（VICReg 损失，3ch）
├── remote_sh/                        # 远程/服务器批处理辅助脚本
│   ├── run_ssc_train_resnet_bat.sh / manage_ssc_train_resnet_bat.sh  # ResNet50+预提取特征 SSC，六数据集×每库 5 轮 best
│   ├── resnet50_batch_result.md      # 上项批量汇总（R1–R5 + mean±std，运行后写入）
│   ├── run_add_ssc_train_vit_bat.sh / manage_add_ssc_train_vit_bat.sh  # add+ViT/Swin 六数据集批量
│   ├── run_add_ssc_train_densenet_bat.sh / manage_add_ssc_train_densenet_bat.sh  # add+DenseNet169 六数据集×3 次
│   ├── run_ssc_train_densenet_bat.sh / manage_ssc_train_densenet_bat.sh  # DenseNet169 SSC 六数据集批量（如启用）
│   ├── run_traditional_train_bat.sh / manage_traditional_train_bat.sh  # 传统线性探针批量
│   ├── densenet_batch_result.md / ssc_densenet169_batch_result.md  # DenseNet 系列批量结果
│   └── *_bat_runner.py               # 由 shell 生成或随仓库提供的启动器
├── MCCFNet/                          # 多通道色彩融合：DenseNet169 + RWP + 线性头（6ch RGB+HSV 端到端）
│   ├── mccfnet_train.py              # 单数据集 / 六数据集 benchmark
│   └── run_mccfnet_train_bat.sh / manage_mccfnet_train_bat.sh
├── ssc_train_resnet_copy.py          # ResNet50 版训练入口（冻结 ResNet 特征 pkl + SSC + 分类头）
├── ssc_train_transformer.py          # Transformer 版训练入口（原版损失）
├── ssc_train_transformer_add.py      # Transformer 版训练入口（add 版损失 + 四路分类头）
├── ssc_train_densnet169_add.py       # DenseNet169-6ch + add 损失 + 内存 GAP 缓存 + EfficientRWPClassifier
├── ieee_ssc_train_resnet.py          # 论文主方法：SSC-ResNet50 五库 benchmark（runs=3，四项指标）
├── ssc_predict.py                    # 推理：view1/view2 余弦相似度统计；含单图推理耗时
├── barlowtwins_train.py              # Barlow Twins 训练；分类器 test 轮次含推理计时
├── SscDataSet_new.py                 # 数据集加载器（当前主用）
├── SscDataSet.py                     # 数据集加载器（旧版）
├── pretrainModels/                   # 本地预训练权重（不提交 git）
├── pretrainFeatures/                 # 预提取特征缓存（不提交 git）
├── model/                            # 训练保存的模型权重（不提交 git）
├── log/                              # 训练日志（不提交 git）
├── data/                             # 数据集根目录（不提交 git）
├── experiment_result/                # 实验汇总表（*.md 多本地、见 .gitignore）与可提交分析脚本
│   └── Webstyle_analysis.py         # WebStyle：按类统计 train / test 图像数（与 remote_sh 中 DATA_ROOT 解析一致）
└── requirements.txt
```

### 仅本地、不提交远程的目录

以下路径在 `.gitignore` 中配置，**克隆仓库后不会存在**，需在本地自行创建或从其他介质拷贝；也不会在 `git push` 时上传到 GitHub / Gitee：

| 路径 | 说明 |
|------|------|
| `GradCAM/` | 本地 Grad-CAM++ 可视化（`gramcam.py` / `gramcam_ours.py` / `gramcam_vit_ssc.py`）、权重与 `output_vit_ssc/` 等（不提交 git） |
| `experiment_result/*.md` | 各实验用 Markdown 汇总表（如 `ours_six_dataset.md`、传统探针结果等，本地生成） |
| `data/`、`model/`、`pretrainFeatures/` 等 | 数据与权重（见上表及 `.gitignore`） |
| `ST-SACLF-ncc_main/models/` | AdaIN 用 `vgg_normalised.pth` 等（本地放置，不提交） |
| `ST-SACLF-ncc_main/experiments/` | AdaIN decoder 权重与可视化输出 |
| `CLIP-based/logs/`、`ST-SACLF-ncc_main/pytorch-AdaIN/logs/` | 批量训练日志与 partial 中间结果 |

---

## Conda 虚拟环境

项目默认使用 Conda 环境 **`ssc`**（Python 3.8 + PyTorch 2.1 + CUDA）。批量脚本会自动尝试 `conda activate ssc`。

### 创建与激活

```bash
# 创建环境（若尚未创建）
conda create -n ssc python=3.8 -y
conda activate ssc

# 安装核心依赖（与 requirements.txt 一致）
pip install -r requirements.txt

# 论文对比实验常用额外包（按需安装）
pip install scikit-learn open-clip-torch tensorboardX imageio protobuf
```

### 环境变量（可选）

| 变量 | 说明 |
|------|------|
| `TORCH_HOME` | timm / torchvision 权重缓存，批量脚本默认指向 `./pretrainModels` |
| `VGG_NORMALISED_PATH` | ST-SACLF AdaIN 用 VGG 权重路径，默认 `ST-SACLF-ncc_main/models/vgg_normalised.pth` |
| `HF_ENDPOINT` | MetaCLIP / HuggingFace 镜像（如 `https://hf-mirror.com`） |

### 验证

```bash
conda activate ssc
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

> **注意：** `ST-SACLF-ncc_main/pytorch-AdaIN/requirements.txt` 为上游 AdaIN 旧版依赖清单，**请勿**据此降级 PyTorch；在 `ssc` 环境下直接运行 `train.py` 即可。

---

## 数据集准备

```
data/<DatasetName>/
├── train/
│   ├── class_1/
│   └── ...
└── test/
    ├── class_1/
    └── ...
```

示例数据位于 `./data/DemoData/`。

五库 benchmark 目录示例：`/mnt/codes/data/style/Painting91/train|test/`，类别子目录为从 **1** 开始的数字文件夹。

---

## 脚本索引

以下按用途分类；带 **`run_*_train_bat.sh`** 的均可后台批量跑五库，配套 **`manage_*_train_bat.sh`** 用于启停、看日志、查看中间/最终结果。

### StyleMaster 主方法（SSC）

| 脚本 | 作用 |
|------|------|
| `ssc_train_transformer.py` | Transformer（Swin/ViT）SSC，VICReg 风格损失 |
| `ssc_train_transformer_add.py` | Transformer SSC，BarlowTwins + SupCon（add 版） |
| `ssc_train_resnet_copy.py` | ResNet50 **预提取 pkl 特征** + SSC + 分类头 |
| `ssc_train_densnet169_add.py` | DenseNet169 **6ch RGB+HSV**，add 损失 + RWP 分类头 |
| `ssc_train_densenet169.py` | DenseNet169 **3ch**，VICReg 损失 + EfficientClassifier |
| `ssc_predict.py` | 推理：view1/view2 余弦相似度统计；含整集推理耗时 |
| `ieee_ssc_train_resnet.py` | 论文 Ours：五库 SSC-ResNet50 benchmark → `ours_multiple.md` |
| `ieee_access_codes/ssc_predict_ieee.py` | 加载 best checkpoint 测试集推理 + failure case → `ssc_failure_case_list.md` |
| `utils/pretrainFeatureExtraction.py` | 提取并缓存 backbone 特征至 `pretrainFeatures/` |

| 批量脚本 | 管理脚本 | 说明 |
|----------|----------|------|
| `remote_sh/run_ssc_train_resnet_bat.sh` | `manage_ssc_train_resnet_bat.sh` | ResNet50 SSC，六库 × 每库 5 轮 best |
| `remote_sh/run_add_ssc_train_vit_bat.sh` | `manage_add_ssc_train_vit_bat.sh` | add + ViT/Swin 六库批量 |
| `remote_sh/run_add_ssc_train_densenet_bat.sh` | `manage_add_ssc_train_densenet_bat.sh` | add + DenseNet169 六库 × 3 次 |
| `remote_sh/run_ssc_train_densenet_bat.sh` | `manage_ssc_train_densenet_bat.sh` | DenseNet169 SSC 六库批量 |

### 自监督对比方法

| 训练脚本 | 批量启动 | 管理 / 查看结果 | 结果文件 |
|----------|----------|-----------------|----------|
| `simclr_train_root.py` | `selfsupervised/run_simclr_train_bat.sh` | `manage_simclr_train_bat.sh` | `ieee_access_paperdata/simclr_multiple.md` |
| `selfsupervised/barlowtwins_train.py` | `run_barlowtwins_train_bat.sh` | `manage_barlowtwins_train_bat.sh` | `ieee_access_paperdata/BarlowTwins_multiple.md` |
| `vicreg_train.py` | `selfsupervised/run_vicreg_train.sh` | `manage_vicreg_train.sh` | `ieee_access_paperdata/vicreg_multiple.md` |

`manage_*` 常用子命令：`start` · `stop` · `status` · `tail` · `logs` · `result` · `help`。前台调试： `./run_*_train_bat.sh fg`。

### CLIP / 视觉-语言基线

| 脚本 | 作用 |
|------|------|
| `CLIP-based/openclip_community_variants_train.py` | OpenCLIP 冻结 ViT-L/14；`linear_probe` / `zero_shot` 两模式，五库 × runs=3 |
| `metaclip_train.py` | MetaCLIP 冻结特征 + 线性分类头 |
| `clip_train.py` | 早期 CLIP 风格分类入口（单数据集） |

| 批量脚本 | 管理脚本 | 结果文件 |
|----------|----------|----------|
| `CLIP-based/run_openclip_train_bat.sh` | `manage_openclip_train_bat.sh` | `ieee_access_paperdata/clip-based_multiple.md` |

本地权重（不提交 git）：`pretrainModels/vit_large_patch16_224.pth`（linear_probe）、`pretrainModels/ViT-L-14-openai.pt`（zero_shot）。

### ST-SACLF（AdaIN）

| 脚本 | 作用 |
|------|------|
| `ST-SACLF-ncc_main/pytorch-AdaIN/train.py` | AdaIN decoder 训练（默认 `max_iter=10000`）+ 冻结 VGG 线性探针，五库 benchmark |
| `ST-SACLF-ncc_main/pytorch-AdaIN/test.py` | AdaIN 风格迁移推理 |

| 批量脚本 | 管理脚本 | 结果文件 |
|----------|----------|----------|
| `run_st_saclf_train_bat.sh` | `manage_st_saclf_train_bat.sh` | `ieee_access_paperdata/ST-SACLF_multiple.md` |

`manage_st_saclf_train_bat.sh` 额外支持 **`partial [Dataset]`**（查看中间 partial 结果）、**`merge`**（重新合并总表）。

本地权重：`ST-SACLF-ncc_main/models/vgg_normalised.pth`（运行前需自行放置）。

### 监督 / 传统 / 去噪基线

| 脚本 | 作用 |
|------|------|
| `traditional_train.py` | 冻结 ImageNet backbone（VGG16 / ResNet50 / ViT-L/16 等）+ 线性探针；支持多数据集 failure case 收集与 test 轮次推理计时 |
| `MCCFNet/mccfnet_train.py` | DenseNet169 + RWP + 6ch RGB+HSV 端到端监督 |
| `denoise/sscae_train.py` | K 路 SCAE 共识 + 分类 |
| `denoise/dae_train.py` | 堆叠 DAE + 分类 |
| `denoise/concurl_train.py` | ConCURL 式投影 MLP + 线性头 |

| 批量脚本 | 管理脚本 | 结果文件 |
|----------|----------|----------|
| `MCCFNet/run_mccfnet_train_bat.sh` | `manage_mccfnet_train_bat.sh` | `ieee_access_paperdata/MCCFNet_multiple.md` |
| `remote_sh/run_traditional_train_bat.sh` | `manage_traditional_train_bat.sh` | `ieee_access_paperdata/vgg16_multiple.md` 等 |

### 其他对比实现（目录内独立入口）

| 方法 | 目录 | 入口 |
|------|------|------|
| Barlow Twins（库） | `barlowtwins/` | `barlowtwins_train.py` |
| SimCLR（库） | `simclr/` | 由 `simclr_train_root.py` 调用 |
| BYOL | `byol/` | `byol_train.py` |
| SimSiam | `simsiam/` | `simsiam_train.py` |
| I-JEPA | `I-JEPA-main/` | `ijepa_train.py` |

---

## 预训练特征提取

训练前需先提取并缓存 backbone 特征：

```bash
python utils/pretrainFeatureExtraction.py
```

提取结果保存至 `./pretrainFeatures/`，训练时通过 `load_dataFeatures()` 加载。

---

## 模型

### Transformer 版（当前主力）

```python
from ssc.Sscreg_transformer import SscReg

model = SscReg(
    backend='swin_base_patch4_window7_224',  # 或 'vit_large_patch16_224'
    input_size=1024,
    output_size=1024,
    depth_projector=3,
)
```

### ResNet 版

```python
from ssc.Sscreg import SscReg

model = SscReg(backend='resnet50', input_size=2048, output_size=2048)
```

### 分类头

| 类名 | 文件 | 结构 | 说明 |
|------|------|------|------|
| `Classifier` | `classifier.py` | Linear×3 + SiLU + Dropout | 基础版 |
| `EfficientClassifier` | `classifier_enhance.py` | 四路拼接（backbone / 残差 / 软正交去噪）+ MLP | Transformer 训练脚本当前所用 |
| `EfficientClassifier` | `classifier_enhance_add.py` | 四路各 256（bb / view1 增强 / view2 增强 / 双视图 MLP）→1024→512→256→cls；无 Dropout | add 版 Transformer 脚本默认 |
| `EfficientRWPClassifier` | `classifier_enhance_add.py` | 与上同四路；融合 head 中 Dropout 换 RegionalWeightedPooling | `ssc_train_densnet169_add.py` 默认 |
| `EfficientClassifier`（IEEE 消融） | `classifier_ieee.py` | 通道相关抑制变体（全通道 / 无抑制 / 随机 / 低相关 / 高相关） | `correlation-based_feature_suppression.py` |
| `StyleEnhancer` | `classifier_enhance_add.py` | 双视图公共风格门控增强（可供实验复用） | — |

---

## 训练

### 原版（VICReg 损失 + 正交化）

```bash
python ssc_train_transformer.py
```

### add 版（BarlowTwins + SupCon 损失 + 四路分类头）

```bash
python ssc_train_transformer_add.py
```

**主要参数（`parameter_load()` 中修改）：**

| 参数 | 原版默认 | add 版默认 |
|------|---------|----------|
| epochs | 20 | 35 |
| batch_size | 128 | 128 |
| offset_batch_size | 512 | 1024 |
| base_lr | 0.001 | 0.001 |
| classifier_iteration | 100 | 100 |
| classifier_lr | 5e-5 | 5e-5 |

训练日志 → `./log/`，最优模型 → `./model/`。

### ResNet50 + 预提取特征（`ssc_train_resnet_copy.py`）

在 **ImageNet ResNet50 预提取特征**（`./pretrainFeatures/{数据集安全名}_resnet50_{train,test}_features.pkl`）上训练 SSC 编码器与分类头；超参默认来自本文件内 `parameter_load()`，也可用命令行覆盖（见 `parse_train_args()`）。

```bash
# 单数据集（示例）
python ssc_train_resnet_copy.py \
  --dataset_name Painting91 \
  --data_root /path/to/style/ \
  --pre_feature_path ./pretrainFeatures \
  --dataset_repeat_runs 1
```

- **`--dataset_repeat_runs`**：同一数据集独立重复完整训练次数；每轮仅统计该轮 **best** 测试准确率，多轮时在日志中输出 `[RUN_BEST]` / `[DATASET_SUMMARY]`（各轮 best 的 **mean±std**）。批量 runner 默认 **5**。
- **WebStyle**：请使用数据子目录 **`webstyle`**（即 `{data_root}/webstyle/train|test`），与 `traditional_train` / `MCCFNet` 等一致；历史路径 `webstyle/subImages` 仍在类别映射中兼容，但批量脚本已改为 `webstyle`。
- 服务器 **六数据集顺序批量**（每库 5 轮，结果表含 R1–R5 与 mean±std）：

```bash
./remote_sh/run_ssc_train_resnet_bat.sh
# 进程与日志：./remote_sh/manage_ssc_train_resnet_bat.sh {status|tail|stop|…}
```

汇总表：`remote_sh/resnet50_batch_result.md`。单机快捷封装可参考 `remote_sh/run_ssc_resnet.sh`。

### add 版 + DenseNet169（6 通道 RGB+HSV，无预提取 pkl）

骨干为 ImageNet DenseNet169 冻结特征（GAP 1664 维），SSC 与分类器阶段使用内存缓存；分类头默认 `EfficientRWPClassifier`。

```bash
python ssc_train_densnet169_add.py
```

**六数据集 × 每库 3 次重复（服务器后台）：**

```bash
./remote_sh/run_add_ssc_train_densenet_bat.sh
# 或
./remote_sh/manage_add_ssc_train_densenet_bat.sh start
```

结果追加至 `remote_sh/densenet_batch_result.md`；进程管理：`manage_add_ssc_train_densenet_bat.sh {status|tail|stop|result}`。

### MCCFNet（端到端监督基线）

`MCCFNet/`：`DenseNet169` + **RegionalWeightedPooling** + 线性分类；输入 6ch（RGB+HSV），与六数据集 benchmark 约定一致。

```bash
python MCCFNet/mccfnet_train.py --data_root <含 train/test 的根目录> --num_classes <K>
# 六数据集依次训练
python MCCFNet/mccfnet_train.py --benchmark_all --data_base /mnt/codes/data/style/
```

批量后台：`./MCCFNet/run_mccfnet_train_bat.sh`（详见 `manage_mccfnet_train_bat.sh`）。

### SimCLR（`simclr_train_root.py`）

ResNet50 SimCLR 自监督预训练 + 线性探针；五库 benchmark，默认 runs=3。

```bash
python simclr_train_root.py --data_root /mnt/codes/data/style/Painting91 --num_classes 13 --runs 3
./selfsupervised/run_simclr_train_bat.sh
./selfsupervised/manage_simclr_train_bat.sh status
```

### OpenCLIP（`CLIP-based/`）

```bash
python CLIP-based/openclip_community_variants_train.py \
  --data_root /mnt/codes/data/style/Painting91 --num_classes 13 \
  --mode linear_probe --runs 3 --dataset_label Painting91
./CLIP-based/run_openclip_train_bat.sh
```

### ST-SACLF AdaIN（`ST-SACLF-ncc_main/pytorch-AdaIN/`）

```bash
cd ST-SACLF-ncc_main/pytorch-AdaIN
python train.py --data_root /mnt/codes/data/style/Painting91 --num_classes 13 --runs 3
# 或项目根目录：
./ST-SACLF-ncc_main/pytorch-AdaIN/manage_st_saclf_train_bat.sh start
./ST-SACLF-ncc_main/pytorch-AdaIN/manage_st_saclf_train_bat.sh partial
./ST-SACLF-ncc_main/pytorch-AdaIN/manage_st_saclf_train_bat.sh result
```

---

## 损失函数

### 原版：`ssc/utils.py` — `criterion()`

$$\mathcal{L} = \lambda_{\text{ortho}} \mathcal{L}_{\text{ortho}} + \lambda_{\text{var}} \mathcal{L}_{\text{var}} + \lambda_{\text{redundancy}} \mathcal{L}_{\text{redundancy}}$$

- **ortho_loss**：两视图 L2 归一化后余弦相似度的平方（默认 $\lambda_{\text{ortho}}=0.5$），驱动方向正交
- **var_loss**：各维标准差下界，防止特征坍缩（默认 $\lambda_{\text{var}}=1.0$）
- **redundancy_loss**：跨视图维度互相关矩阵的非对角项惩罚（默认 $\lambda_{\text{redundancy}}=0.1$），抑制跨维冗余

（实现中**无**单独 MSE 视图不变项；与 acc 的关系可在 add 版中对比。）

### add 版：`ssc/utils_add.py` — `criterion_align()`

$$\mathcal{L} = \lambda_{\text{align}} \cdot \mathcal{L}_{\text{BT}} + \lambda_{\text{var}} \cdot \mathcal{L}_{\text{var}} + \lambda_{\text{sc}} \cdot \mathcal{L}_{\text{SupCon}}$$

- **BarlowTwins**：对角互相关趋近 1，驱动两视图对齐公共风格
- **var_loss**：防止特征坍缩
- **SupCon**：有监督对比损失，同类样本拉近、异类推远，保证对齐特征的判别性

---

## 推理

```bash
# SSC 编码器：view1/view2 余弦相似度 + 推理耗时
python ssc_predict.py

# IEEE 论文：SSC-ResNet50 测试集准确率 + failure case + 推理耗时
python ieee_access_codes/ssc_predict_ieee.py
python ieee_access_codes/ssc_predict_ieee.py --datasets Painting91 FashionStyle14
```

---

## IEEE Access 论文实验（`ieee_code_done`）

论文复现相关脚本与产出集中在 `ieee_access_codes/` 与 `ieee_access_paperdata/`。

### 主方法训练与 benchmark

```bash
# 五数据集 SSC-ResNet50（runs=3，Acc / Macro-F1 / Weighted-F1 / Balanced Acc）
python ieee_ssc_train_resnet.py --benchmark_all

# 单数据集
python ieee_ssc_train_resnet.py --dataset_name Painting91 --runs 3

# 后台批量
./ieee_access_codes/manage_ieee_ssc_train_bat.sh start
```

结果写入 `ieee_access_paperdata/ours_multiple.md`。最佳 checkpoint 默认存放于 `ieee_access_paperdata/models/`（本地，不提交 git）。

### 测试集推理与错误案例分析

```bash
# SSC 预测错误 → ssc_failure_case_list.md
python ieee_access_codes/ssc_predict_ieee.py

# ViT-L/16 传统探针错误案例（Painting91 + FashionStyle14，runs=1）
python traditional_train.py \
  --backbone vit_l_16 \
  --benchmark_datasets Painting91 FashionStyle14 \
  --runs 1 \
  --failure_md ieee_access_paperdata/ivt_failure_case_list.md
```

### 定性可视化候选（ViT 错 / SSC 对）

`ieee_access_paperdata/qualitative_vis_list.md` 由 `ivt_failure_case_list.md` 与 `ssc_failure_case_list.md` 交叉筛选得到。本地 Grad-CAM 对比（不提交 git）：

```bash
python GradCAM/gramcam_vit_ssc.py
# 输出：GradCAM/output_vit_ssc/（原图 | ViT | SSC 对比图）
```

### 统计检验与消融

```bash
python ieee_access_codes/ieee_bootstrap.py
python ieee_access_codes/correlation-based_feature_suppression.py
python ieee_access_codes/plot_ieee_ssc_confusion_matrix.py
```

对应结果：`ieee_bootstrap.md`、`correlation-based_feature_suppression.md`、`ours_ssc_confusion_matrix_*.png`。

### 传统 ViT 探针（含推理计时）

```bash
python traditional_train.py --backbone vit_l_16 --data_root /mnt/codes/data/style/Painting91 --runs 1
```

test 轮次（每 10 iter）日志输出 `inference_time` 与 `ms/image`（端到端：backbone + 分类头）。

---

## 实验结果与数据分析

### 论文汇总表（`ieee_access_paperdata/`）

各对比方法在五数据集上 **runs=3** 的四项指标（Accuracy / Macro-F1 / Weighted-F1 / Balanced Accuracy），格式统一，含 run1–run3 与 mean±std：

| 文件 | 对应方法 |
|------|----------|
| `ours_multiple.md` | **Ours（SSC-ResNet50）** |
| `simclr_multiple.md` | SimCLR |
| `BarlowTwins_multiple.md` | Barlow Twins |
| `MCCFNet_multiple.md` | MCCFNet |
| `clip-based_multiple.md` | OpenCLIP（linear_probe / zero_shot） |
| `ST-SACLF_multiple.md` | ST-SACLF AdaIN |
| `vgg16_multiple.md` / `resnet50_multiple.md` / `vit_l_16_multiple.md` | 传统线性探针 |

### 错误案例与定性分析

| 文件 | 说明 |
|------|------|
| `ssc_failure_case_list.md` | SSC-ResNet50 测试集预测错误（含推理时间） |
| `ivt_failure_case_list.md` | ViT-L/16 传统探针测试集预测错误 |
| `qualitative_vis_list.md` | ViT 判错且 SSC 判对的样本清单（Grad-CAM 候选） |
| `ieee_bootstrap.md` | Bootstrap 显著性检验 |
| `correlation-based_feature_suppression.md` | 通道相关抑制消融 |

训练进行中可通过各 `manage_*_train_bat.sh result` 或 ST-SACLF 的 `partial` 查看增量结果。

### 数据分析脚本

- **`experiment_result/Webstyle_analysis.py`**：在 `train` / `test` 等划分下，按类统计 `webstyle` 数据集中各子类图像数量；默认从 `remote_sh/run_ssc_train_resnet_bat.sh` 内嵌的 `DATA_ROOT` 与批量训练一致，也可用 `--data_root` 显式指定。

```bash
python experiment_result/Webstyle_analysis.py
# python experiment_result/Webstyle_analysis.py --data_root /path/to/style/
```

- 同目录下 `*.md` 多为本地复现或批量评测汇总，默认通过 `.gitignore` 不提交；需要共享时再单独处理。

---

## 对比方法

> 完整脚本说明见上文 **[脚本索引](#脚本索引)**。

| 方法 | 目录 | 论文 benchmark 入口 |
|------|------|---------------------|
| SimCLR | `simclr/` + `simclr_train_root.py` | `selfsupervised/run_simclr_train_bat.sh` |
| Barlow Twins | `barlowtwins/` + `selfsupervised/barlowtwins_train.py` | `selfsupervised/run_barlowtwins_train_bat.sh` |
| OpenCLIP | `CLIP-based/` | `CLIP-based/run_openclip_train_bat.sh` |
| ST-SACLF AdaIN | `ST-SACLF-ncc_main/pytorch-AdaIN/` | `run_st_saclf_train_bat.sh` |
| MCCFNet | `MCCFNet/` | `MCCFNet/run_mccfnet_train_bat.sh` |
| 传统线性探针 | 根目录 `traditional_train.py` | `remote_sh/run_traditional_train_bat.sh` |
| BYOL | `byol/` | `byol_train.py` |
| SimSiam | `simsiam/` | `simsiam_train.py` |
| I-JEPA | `I-JEPA-main/` | `ijepa_train.py` |
| MetaCLIP | 根目录 | `metaclip_train.py` |

### 自监督五数据集批量（`selfsupervised/`）

默认 **runs=3**，结果写入 `ieee_access_paperdata/*.md`；`manage_*_train_bat.sh` 支持 `start` / `stop` / `status` / `tail` / `result`。前台调试：`./run_*_train_bat.sh fg`。

---

## denoise：冻结主干上的风格分类基线

以下脚本与 `traditional_train.py` 共用同一套 **冻结 ImageNet 预训练 backbone**（`build_backbone` + 一次特征缓存），在 **train/test 的 ImageFolder** 上训练轻量头并报告 **test 准确率**。六数据集批量评测时路径与类别数与 `remote_sh` 中约定一致（`Painting91`、`Pandora`、`AVAstyle`、`FashionStyle14`、`Arch`、`webstyle` 等）。

| 脚本 | 模型要点 | 常用命令 |
|------|----------|----------|
| `denoise/sscae_train.py` | `CSCAE`：K 路 `SCAE` + 共识 latent + 分类损失 | `python denoise/sscae_train.py --benchmark_all` |
| `denoise/dae_train.py` | `SDAEClassifier`：两层堆叠 DAE + 预训练 + 微调 | `python denoise/dae_train.py --benchmark_all` |
| `denoise/concurl_train.py` | `ConCURLClassifier`：ProjectionMLP（L2）+ 线性分类 | `python denoise/concurl_train.py --benchmark_all` |

**公共参数（示例）：** `--data_root` / `--num_classes`（单数据集）；`--benchmark_all` + `--data_base`（默认 `/mnt/codes/data/style/`）；`--backbone`（默认 `vgg16`）；`--run` / `--runs`（重复次数与 mean±std，默认 3）；`--result_md`（结果 Markdown 路径）。

**传统线性探针（多 backbone 对比）：**

```bash
python traditional_train.py --backbone vit_l_16 --data_root /mnt/codes/data/style/Painting91 --runs 1
python traditional_train.py --backbone vit_l_16 --benchmark_datasets Painting91 FashionStyle14 --runs 1
```

---

## 版本与远程仓库

| 远程 | 地址 | 说明 |
|------|------|------|
| GitHub | `git@github.com:cuijia1247/StyleMaster.git` | 主远程（`github`） |
| Gitee | `git@gitee.com:cuijia_1247/SubStyleClassfication.git` | 备用（`origin`） |

检出 IEEE 代码完成版本：

```bash
git clone git@github.com:cuijia1247/StyleMaster.git
cd StyleMaster
git checkout ieee_code_done
```

---

## Citation

```
waiting for our new released paper citation
```
