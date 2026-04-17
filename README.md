# SSC — Sub-Style Classification

**Author:** cuijia1247 | **Started:** Oct. 2024 | **Current Version:** Apr. 2026

> Paper coming soon.

---

## 项目简介

StyleMaster 是一个面向风格的特征学习框架，由两部分组成：
- **Style Consensus Learning (SCL)**：主风格特征学习
- **Sub-Style Classification (SSC)**：子风格细粒度分类

SSC 核心思想：将同一幅画的两个随机裁剪子图（view1 / view2）送入 SSC 编码器，利用自监督损失约束特征空间，再训练轻量分类头完成风格判别。

支持数据集：`Painting91` · `AVAstyle` · `WikiArt3` · `FashionStyle14` · `Pandora` · `Arch` · `artbench`

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
│   ├── Sscreg_densenet169.py         # DenseNet169 冻结骨干 + 6ch 投影 SSC 编码器（1664→1664）
│   ├── classifier_original.py        # 原始分类头存档
│   ├── utils.py                      # 原版损失函数（VICReg + 正交化）
│   └── utils_add.py                  # add 版损失函数（BarlowTwins + SupCon）
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
├── selfsupervised/                   # SimCLR / Barlow Twins 等自监督基线（六数据集 benchmark）
│   ├── simclr_train.py               # SimCLR 预训练 + 线性探针
│   ├── barlowtwins_train.py         # Barlow Twins + 线性探针
│   ├── run_*_train_bat.sh           # nohup 批量训练（SSH 断连可续跑）
│   ├── manage_*_train_bat.sh        # 启停 / tail 日志 / 查看结果表
│   ├── logs/                         # 批量运行日志（本地，默认不提交）
│   └── *_result.md                   # 评测汇总表（本地生成时可不提交）
├── remote_sh/                        # 远程/服务器批处理辅助脚本
│   ├── run_add_ssc_train_vit_bat.sh / manage_add_ssc_train_vit_bat.sh  # add+ViT/Swin 六数据集批量
│   ├── run_add_ssc_train_densenet_bat.sh / manage_add_ssc_train_densenet_bat.sh  # add+DenseNet169 六数据集×3 次
│   ├── run_traditional_train_bat.sh / manage_traditional_train_bat.sh  # 传统线性探针批量
│   ├── densenet_batch_result.md      # DenseNet 批量实验汇总（运行后追加）
│   └── *_bat_runner.py               # 由 shell 生成或随仓库提供的启动器
├── MCCFNet/                          # 多通道色彩融合：DenseNet169 + RWP + 线性头（6ch RGB+HSV 端到端）
│   ├── mccfnet_train.py              # 单数据集 / 六数据集 benchmark
│   └── run_mccfnet_train_bat.sh / manage_mccfnet_train_bat.sh
├── ssc_train_resnet.py               # ResNet 版训练入口
├── ssc_train_transformer.py          # Transformer 版训练入口（原版损失）
├── ssc_train_transformer_add.py      # Transformer 版训练入口（add 版损失 + 四路分类头）
├── ssc_train_densnet169_add.py       # DenseNet169-6ch + add 损失 + 内存 GAP 缓存 + EfficientRWPClassifier
├── ssc_predict.py                    # 推理：计算 view1/view2 余弦相似度统计
├── SscDataSet_new.py                 # 数据集加载器（当前主用）
├── SscDataSet.py                     # 数据集加载器（旧版）
├── pretrainModels/                   # 本地预训练权重（不提交 git）
├── pretrainFeatures/                 # 预提取特征缓存（不提交 git）
├── model/                            # 训练保存的模型权重（不提交 git）
├── log/                              # 训练日志（不提交 git）
├── data/                             # 数据集根目录（不提交 git）
└── requirements.txt
```

---

## 安装

**环境要求：**
- Python 3.8.19
- PyTorch 2.1.0 + torchvision 0.16.0
- CUDA（推荐）

```bash
pip install -r requirements.txt
```

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

批量后台：`./MCCFNet/run_mccfnet_train_bat.sh`（详见同目录 `manage_mccfnet_train_bat.sh`）。

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
python ssc_predict.py
```

---

## 对比方法

| 方法 | 目录 | 入口 |
|------|------|------|
| Barlow Twins | `barlowtwins/` | `barlowtwins_train.py` |
| SimCLR | `simclr/` | `simclr_train.py` |
| BYOL | `byol/` | `byol_train.py` |
| SimSiam | `simsiam/` | `simsiam_train.py` |
| I-JEPA | `I-JEPA-main/` | `ijepa_train.py` |

### 自监督六数据集批量（`selfsupervised/`）

与 `traditional_train.py` 相同的数据根目录与类别数约定，默认 **`--benchmark_all --runs 3`**，结果追加到对应 `*_result.md`。适合服务器 **nohup** 后台，断开 SSH 仍可继续。

| 任务 | 训练脚本 | 后台启动 | 进程与日志管理 |
|------|----------|----------|----------------|
| SimCLR | `selfsupervised/simclr_train.py` | `./selfsupervised/run_simclr_train_bat.sh` | `./selfsupervised/manage_simclr_train_bat.sh {start\|stop\|tail\|result\|…}` |
| Barlow Twins | `selfsupervised/barlowtwins_train.py` | `./selfsupervised/run_barlowtwins_train_bat.sh` | `./selfsupervised/manage_barlowtwins_train_bat.sh …` |

单次前台调试可加参数：`./selfsupervised/run_simclr_train_bat.sh fg`。

---

## denoise：冻结主干上的风格分类基线

以下脚本与 `traditional_train.py` 共用同一套 **冻结 ImageNet 预训练 backbone**（`build_backbone` + 一次特征缓存），在 **train/test 的 ImageFolder** 上训练轻量头并报告 **test 准确率**。六数据集批量评测时路径与类别数与 `remote_sh` 中约定一致（`Painting91`、`Pandora`、`AVAstyle`、`FashionStyle14`、`Arch`、`webstyle` 等）。

| 脚本 | 模型要点 | 常用命令 |
|------|----------|----------|
| `denoise/sscae_train.py` | `CSCAE`：K 路 `SCAE` + 共识 latent + 分类损失 | `python denoise/sscae_train.py --benchmark_all` |
| `denoise/dae_train.py` | `SDAEClassifier`：两层堆叠 DAE + 预训练 + 微调 | `python denoise/dae_train.py --benchmark_all` |
| `denoise/concurl_train.py` | `ConCURLClassifier`：ProjectionMLP（L2）+ 线性分类 | `python denoise/concurl_train.py --benchmark_all` |

**公共参数（示例）：** `--data_root` / `--num_classes`（单数据集）；`--benchmark_all` + `--data_base`（默认 `/mnt/codes/data/style/`）；`--backbone`（默认 `vgg16`）；`--run` / `--runs`（重复次数与 mean±std，默认 3）；`--result_md`（结果 Markdown 路径）。

**传统线性探针（多 backbone 对比）：** `python traditional_train.py --backbone resnet50 --data_root <数据集根目录>`。

---

## Citation

```
waiting for our new released paper citation
```
