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
│   ├── classifier_enhance_add.py     # add 系列三路融合分类头（路1:256 + 路2:512 + 路3:256）
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
├── ssc_train_resnet.py               # ResNet 版训练入口
├── ssc_train_transformer.py          # Transformer 版训练入口（原版损失）
├── ssc_train_transformer_add.py      # Transformer 版训练入口（add 版损失 + 新分类头）
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
| `EfficientClassifier` | `classifier_enhance.py` | 多路投影 + LayerNorm + GELU | 增强版（原版） |
| `EfficientClassifier` | `classifier_enhance_add.py` | 三路融合（256+512+256→1024）+ StyleEnhancer | add 版（当前实验） |

---

## 训练

### 原版（VICReg 损失 + 正交化）

```bash
python ssc_train_transformer.py
```

### add 版（BarlowTwins + SupCon 损失 + StyleFusion 分类头）

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

---

## 损失函数

### 原版：`ssc/utils.py` — `criterion()`

$$\mathcal{L} = \mathcal{L}_{\text{var}} + \mathcal{L}_{\text{invar}} + \lambda_{\text{ortho}} \cdot \mathcal{L}_{\text{ortho}}$$

- **var_loss**：防止特征坍缩
- **invar_loss**：MSE 约束两视图一致
- **ortho_loss**：余弦相似度²，驱动两视图正交（已发现与 acc 负相关，add 版已去除）

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

---

## Citation

```
waiting for our new released paper citation
```
