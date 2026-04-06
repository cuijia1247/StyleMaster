# Research Log — Sub-Style Classification

---

## 已确定最优参数

| 模型 | 参数配置 | Painting91 最优 Acc |
|------|---------|-------------------|
| ResNet-50 | epochs=200, base_lr=0.009, cls_iteration=200 | **76.05%** |
| ViT-L/16 | epochs=100, base_lr=0.009, cls_iteration=200 | **73.95%** |
| Swin-Base（原版损失） | epochs=20, base_lr=0.001, cls_iteration=100 | **~73.3%** |
| Swin-Base（add 版损失） | epochs=35, base_lr=0.001, cls_iteration=100, offset_bs=1024 | **~72.7%**（持续实验中）|

---

## 实验记录

### 20260403
- ResNet-50 最优参数确定：epochs=200, base_lr=0.009, cls_iteration=200，best acc=76.05%
- ViT-L/16 最优参数确定：epochs=100, base_lr=0.009, cls_iteration=200，best acc=73.95%

### 20260406 — SSC 损失函数改进实验（Painting91）

**背景**：原版 `criterion()` 含正交化损失（`lam_ortho`），实验发现其与分类准确率负相关。

#### 实验 A（基线）：`lam_ortho=0.1, lam_redundancy=0.05`（log: 12-17-06）
- iter0 全局最高 acc：**0.7143**
- iter1 最高 acc：**0.7332**
- iter2 最高 acc：**0.7185**
- 特点：随迭代轮次增加 acc 稳步提升

#### 实验 B：`lam_ortho=0.05, lam_redundancy=0.1`（log: 14-02-51）
- iter0 最高：0.7143（持平）
- iter1 最高：**0.7227**（↓ vs A）
- 结论：降低 ortho 权重、增大 redundancy 无正向效果，iter1 退步约 1%

#### 实验 C（add 版 v1）：去除 ortho，改用 BarlowTwins 对齐损失（log: 15-09-49）
- 分类器：`EfficientClassifier`（三路 256+256+256→768，路3为 ssc_common）
- iter0 最高：**0.7164**（微升）
- iter1 最高：0.7227（vs A 仍低）
- SSC train loss 起点高（1.808），仍在下降，有潜力
- 问题：off_diag 归一化系数为 D²，实际惩罚近似为 0

#### 实验 D（add 版 v2）：off_diag 归一化修复 + 三路结构调整（log: 16-25-43）
- 修改：off_loss 除以 `D*(D-1)` 而非 `D²`；分类头改为 256+512+256 三路；offset_bs=1024；epochs=35
- iter0 全局最高：**0.7269**（↑ vs 所有之前实验）
- iter1 最高：**0.7164**（漂移期退步）
- iter2 最高：**0.7143**（持续下滑）
- 根因：BarlowTwins 驱动特征对齐公共风格，但无判别性约束，acc 随 SSC loss 下降而退步

#### 实验 E（add 版 v3）：引入 SupCon 有监督对比损失（log: 18-22-35）
- 新增 `supcon_loss(x, y, labels, temperature=0.5)`，权重 `lam_supcon=0.5`
- 遭遇 NaN：`log_softmax` 输出 `-inf` 与 `mask_pos=0` 相乘，`-inf × 0 = NaN`
- 修复：用 `torch.where(mask_pos, log_prob, zeros)` 替换直接乘法

#### 实验 F（add 版 v3 修复后）：BarlowTwins + SupCon（当前进行中）
- 分类器每次从零重新初始化（去除热启动，避免过拟合）
- 新增 `CosineAnnealingLR` 给分类器，`es_patience` 从 12 降至 7
- 待观察 acc 趋势

---

## TODO

- [ ] 观察 add 版 v3（SupCon）实验结果，对比实验 A 基线
- [ ] 若 SupCon 有效，尝试调整 `lam_supcon`（0.2 / 0.5 / 1.0）和 `temperature`（0.3 / 0.5 / 0.7）
- [ ] image patch selection 策略优化
- [ ] image patch size 敏感性分析
- [ ] 更多 backbone 特征对比（ResNet-50 / ViT-L / Swin-B / Swin-L）
- [ ] 消融实验设计（var / align / supcon 各项贡献）
- [ ] 在其他数据集上运行（AVAstyle / WikiArt3 / FashionStyle14）
