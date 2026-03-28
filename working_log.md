# 项目工作日志

## 2026-03-28 项目清理

### 脚本清单

#### SSC 核心脚本（当前有效）

| 脚本 | 功能 |
|------|------|
| `ssc_train.py` | SSC ResNet50 版本主训练脚本，含分类器联合训练 |
| `ssc_train_transformer.py` | SSC Transformer（Swin/ViT）版本主训练脚本 |
| `ssc_predict.py` | 对指定数据集推理，统计 view1/view2 余弦相似度分布 |
| `SscDataSet.py` | SSC 数据集加载类，被各训练脚本导入使用 |
| `visualize_sscreg_transformer.py` | 可视化 SSCReg Transformer 模型特征输出 |

#### 基线对比模型训练脚本

| 脚本 | 方法 |
|------|------|
| `barlowtwins_train.py` | Barlow Twins 对比学习 |
| `byol_train.py` | BYOL 自监督学习 |
| `simclr_train.py` | SimCLR 对比学习 |
| `simsiam_train.py` | SimSiam 自监督学习 |
| `ijepa_train.py` | I-JEPA 联合嵌入预测架构 |

---

### 问题记录与处理状态

#### 1. 无效/废弃脚本

| 脚本 | 原因 | 状态 |
|------|------|------|
| `SC_train.py` | 头部注明"DO NOT USE IT NOW"，旧版训练脚本 | ✅ 已删除 |
| `SC_train_optimization.py` | 同上，旧版参数搜索脚本 | ✅ 已删除 |
| `SC_predict.py` | 头部注明"未完成"，功能不完整 | ✅ 已删除 |
| `Logging_test.py` | 18 行临时测试文件，无实际功能 | ✅ 已删除 |

#### 2. 功能重复脚本

| 重复组 | 说明 | 状态 |
|--------|------|------|
| `SC_train.py` / `SC_train_optimization.py` vs `ssc_train.py` | 旧版与新版重复 | ✅ 旧版已删除 |
| `SC_predict.py` / `SC_predict_HR.py` vs `ssc_predict.py` | 三版预测脚本重复 | ✅ 旧版已清理 |
| `manage_ssc_*.sh` vs `run_ssc_*_background.sh` | manage 脚本内嵌 run 脚本功能，层叠重复 | ✅ 已随脚本整体清理 |

#### 3. Shell 脚本 Bug

| 问题 | 描述 | 状态 |
|------|------|------|
| `manage_ssc_vit.sh` 错误引用 | `start_training` 调用不存在的 `./run_ssc_background.sh`，正确名为 `run_ssc_vit_background.sh` | ✅ 随脚本清理消除 |

#### 4. HR 作者脚本归档

| 操作 | 状态 |
|------|------|
| 将 `SC_predict_HR.py`、`SC_clean_HR.py`、`gram_threshold_HR.py`、`create_train_data_HR.py` 移至 `HR/` 文件夹 | ✅ 已完成 |

#### 5. .gitignore 补充

| 新增排除规则 | 状态 |
|-------------|------|
| `log/`、`log_backup/`、`prediction/` | ✅ 已添加 |
| `model/`、`model_old/`、`pretrainModels/`、`HR/` | ✅ 已添加 |
| `*.pid`（运行时 PID 文件） | ✅ 已添加 |

---

### GitHub 推送记录

| 时间 | 操作 |
|------|------|
| 2026-03-28 | 初次推送至 `git@github.com:cuijia1247/StyleMaster.git` |
| 2026-03-28 | `master` 分支合并入 `main`，远程 `master` 已删除 |
