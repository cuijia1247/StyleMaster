# SSC — Sub-Style Classification

**Author:** cuijia1247 | **Started:** Oct. 2024 | **Current Version:** Mar. 2026

> Paper coming soon.

---

## 项目简介

StyleMaster是一个面向风格的特征学习框架。其包括主风格特征(Style Consensus Learning)和子风格特征(Sub-Style Learning)两部分组成

Sub-Style Learning是一个基于自监督对比学习的艺术风格细粒度分类框架。核心思想：将同一幅画的两个随机裁剪子图（view1 / view2）送入 SSC 编码器，利用SSLM风格的损失函数约束特征空间，再训练轻量分类头完成风格判别。

支持的数据集：`Painting91` · `AVAstyle` · `WikiArt3` · `FashionStyle14` · `Pandora` · `Arch` · `artbench`

---

## 目录结构

```
StyleMaster/
├── ssc/                              # SSC 核心模块
│   ├── Sscreg.py                     # ResNet-based SscReg 模型
│   ├── Sscreg_transformer.py         # Transformer-based SscReg 模型（Swin / ViT）
│   ├── Sscreg_flexible.py            # 灵活配置版本
│   ├── Backend.py                    # ResNet backbone 封装
│   ├── classifier.py                 # 分类头（Classifier / EfficientClassifier / AdvancedClassifier）
│   └── utils.py                      # 损失函数、数据增强变换
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
├── HR/                               # HR 作者贡献脚本（数据清洗、阈值分析等）
├── ssc_train.py                      # ResNet 版训练入口
├── ssc_train_transformer.py          # Transformer 版训练入口
├── ssc_predict.py                    # 推理：计算 view1/view2 余弦相似度统计
├── SscDataSet.py                     # 数据集加载器
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
- CUDA（推荐，GPU 加速）

```bash
pip install -r requirements.txt
```

主要依赖还包括 `timm`（Swin/ViT backbone）、`einops`、`pytorch_lightning`、`scipy`。

---

## 数据集准备

按以下结构组织数据集：

```
data/<DatasetName>/
├── train/
│   ├── class_1/
│   ├── class_2/
│   └── ...
└── test/
    ├── class_1/
    ├── class_2/
    └── ...
```

示例数据位于 `./data/DemoData/`。

---

## 预训练特征提取

训练前需先用骨干网络提取并缓存特征，避免每轮重复推理：

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
    pretrained_backend=True  # 从 pretrainModels/ 加载本地权重
)
```

### ResNet 版

```python
from ssc.Sscreg import SscReg

model = SscReg(
    backend='resnet50',
    input_size=2048,
    output_size=2048,
    depth_projector=3,
    pretrained_backend=False
)
```

### 分类头

| 类名 | 结构 | 适用场景 |
|------|------|----------|
| `Classifier` | Linear × 3 + SiLU + Dropout | 基础版 |
| `EfficientClassifier` | Linear-BN-ReLU-Dropout × 4 | 推荐（当前默认） |
| `AdvancedClassifier` | 多头注意力 + 残差块 + MLP | 高容量版 |

---

## 训练

### ResNet 版

```bash
python ssc_train.py
```

### Transformer 版

```bash
python ssc_train_transformer.py
```

**主要训练参数（在各脚本 `parameter_load()` 中修改）：**

| 参数 | ResNet 默认 | Transformer 默认 |
|------|-------------|-----------------|
| epochs | 200 | 120 |
| batch_size | 64 | 16 |
| base_lr | 0.008 | 0.0001 |
| image_size | 64 | 224 |
| classifier_iteration | 200 | 1000 |

训练日志保存至 `./log/`，最优模型保存至 `./model/`（含精度信息的文件名）。

---

## 损失函数

`ssc/utils.py` 中的 `criterion()` 实现 VICReg 风格三项损失：

```
Loss = var_loss + invar_loss + cross_loss
```

- **var_loss**：保持特征多样性，防止坍缩
- **invar_loss**：MSE 约束两视图特征一致
- **cross_loss**：交叉相关矩阵去相关（Barlow Twins 思路）

---

## 推理

```bash
python ssc_predict.py
```

遍历数据集全部分片（train / test / val），计算 view1 与 view2 的余弦相似度并输出统计信息（均值、方差、标准差、最大/最小值）。

---

## 对比方法

项目包含以下 SSL 方法的对比实现，训练入口分别为 `*_train.py`：

- **Barlow Twins** (`barlowtwins/`)
- **SimCLR** (`simclr/`)
- **BYOL** (`byol/`)
- **SimSiam** (`simsiam/`)
- **I-JEPA** (`I-JEPA-main/`)

---

## Citation

```
waiting for our new released paper citation
```
