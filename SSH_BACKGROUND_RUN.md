# SSH 后台运行 SSC Transformer 训练指南

## 📋 概述

本指南说明如何通过SSH在conda虚拟环境`vicreg`下后台运行`ssc_train_transformer.py`脚本，确保即使断开SSH连接，训练程序仍能继续运行。

**🆕 更新说明**: 脚本现在使用本地预训练模型，无需网络连接，完全离线运行。

## 🚀 快速开始

### 1. 启动后台训练

```bash
# 通过SSH连接到服务器后，运行以下命令：
cd /home/cuijia1247/Codes/SubStyleClassfication
./run_ssc_background.sh
```

### 2. 查看训练状态

```bash
# 查看训练是否正在运行
./manage_ssc.sh status
```

### 3. 查看实时日志

```bash
# 实时查看训练日志
./manage_ssc.sh tail
```

### 4. 停止训练

```bash
# 停止正在运行的训练
./manage_ssc.sh stop
```

## 📁 文件说明

- `run_ssc_background.sh` - 后台运行脚本（支持本地模型加载）
- `manage_ssc.sh` - 训练管理脚本
- `ssc_train_transformer.py` - 主训练脚本（已配置使用cuda:1，支持本地模型）
- `pretrainModels/vit_small_patch16_224.pth` - 本地预训练模型文件

## 🔧 详细使用方法

### 启动训练

```bash
# 方法1: 直接运行脚本
./run_ssc_background.sh

# 方法2: 使用管理脚本启动
./manage_ssc.sh start

# 方法3: 使用nohup命令（备选方案）
nohup bash run_ssc_background.sh > startup.log 2>&1 &
```

### 管理训练

```bash
# 查看训练状态
./manage_ssc.sh status

# 查看最新日志（最后50行）
./manage_ssc.sh logs

# 实时查看日志
./manage_ssc.sh tail

# 列出所有相关进程
./manage_ssc.sh list

# 停止训练
./manage_ssc.sh stop

# 重启训练
./manage_ssc.sh restart

# 清理日志文件
./manage_ssc.sh clean

# 显示帮助
./manage_ssc.sh help
```

## 📊 日志文件

- 训练日志保存在: `./log/ssc_background_YYYY-MM-DD-HH-MM-SS.log`
- 进程ID保存在: `./ssc_train.pid`

## 🔍 监控训练进度

### 1. 查看GPU使用情况
```bash
# 查看GPU状态
nvidia-smi

# 持续监控GPU使用
watch -n 1 nvidia-smi
```

### 2. 查看系统资源使用
```bash
# 查看CPU和内存使用
htop

# 查看磁盘使用
df -h
```

### 3. 查看训练日志
```bash
# 查看最新日志
tail -f ./log/ssc_background_*.log

# 搜索特定内容
grep "accuracy" ./log/ssc_background_*.log
```

## ⚠️ 注意事项

1. **环境要求**: 确保conda环境`vicreg`已正确安装所有依赖
2. **GPU配置**: 脚本已配置使用`cuda:1`设备
3. **存储空间**: 确保有足够的磁盘空间存储模型和日志
4. **本地模型**: 确保 `./pretrainModels/vit_small_patch16_224.pth` 文件存在
5. **离线运行**: 脚本完全离线运行，无需网络连接
6. **模型兼容性**: 本地模型已适配为backbone特征提取器

## 🛠️ 故障排除

### 本地模型文件问题
如果遇到模型文件相关错误：

```bash
# 检查本地模型文件是否存在
ls -la ./pretrainModels/vit_small_patch16_224.pth

# 如果文件不存在，请确保模型文件已正确放置
# 模型文件应该位于: ./pretrainModels/vit_small_patch16_224.pth
```

### 训练无法启动
```bash
# 检查conda环境
conda activate vicreg
python -c "import torch; print(torch.cuda.is_available())"

# 检查脚本权限
ls -la run_ssc_background.sh manage_ssc.sh

# 检查本地模型文件
ls -la pretrainModels/vit_small_patch16_224.pth
```

### 训练意外停止
```bash
# 查看错误日志
./manage_ssc.sh logs

# 检查系统资源
free -h
df -h
```

### 无法停止训练
```bash
# 强制停止
pkill -f ssc_train_transformer.py

# 或者使用进程ID
kill -9 $(cat ssc_train.pid)
```

## 📞 支持

如果遇到问题，请检查：
1. conda环境是否正确激活
2. 所有依赖包是否已安装
3. GPU是否可用
4. 磁盘空间是否充足
5. 本地模型文件是否存在 (`./pretrainModels/vit_small_patch16_224.pth`)
6. 日志文件中的错误信息

## 🔄 更新日志

- **2024-10-19**: 创建SSH后台运行脚本和管理工具
- **2024-10-19**: 支持本地模型加载，移除网络依赖
- **2024-10-19**: 添加完整的训练管理功能
- **2024-10-19**: 修复网络下载bug，实现完全离线运行
- **2025-10-19**: 将Swin Transformer替换为ViT-small/16模型
