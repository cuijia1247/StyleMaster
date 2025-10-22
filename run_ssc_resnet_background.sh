#!/bin/bash

# SSH 后台运行 SSC ResNet 训练脚本
# Author: cuijia1247
# Date: 2025-10-22
# 使用 ResNet50 作为骨干网络进行训练

# 设置脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 创建日志目录
mkdir -p log

# 生成时间戳
TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S")
LOG_FILE="log/ssc_resnet_${TIMESTAMP}.log"
PID_FILE="ssc_resnet_train.pid"

# 检查是否已有训练进程在运行
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "ResNet训练进程已在运行 (PID: $OLD_PID)"
        echo "如需重新启动，请先运行: kill $OLD_PID && rm $PID_FILE"
        exit 1
    else
        echo "清理旧的PID文件"
        rm -f "$PID_FILE"
    fi
fi

# 检查conda环境
if ! command -v conda &> /dev/null; then
    echo "错误: conda未安装或未在PATH中"
    exit 1
fi

# 激活conda环境
echo "激活conda环境: vicreg"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate vicreg

# 检查环境是否正确激活
if [ "$CONDA_DEFAULT_ENV" != "vicreg" ]; then
    echo "错误: 无法激活conda环境 'vicreg'"
    echo "请确保环境存在: conda env list"
    exit 1
fi

# 检查ResNet预训练特征文件（如果需要的话）
# FEATURE_FILE="pretrainFeatures/Painting91_resnet50_train.pkl"
# if [ ! -f "$FEATURE_FILE" ]; then
#     echo "警告: ResNet预训练特征文件不存在: $FEATURE_FILE"
#     echo "如果需要使用预训练特征，请先运行特征提取脚本"
# fi

# 检查GPU可用性
echo "检查GPU状态..."
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'CUDA设备数量: {torch.cuda.device_count()}'); print(f'当前设备: {torch.cuda.current_device() if torch.cuda.is_available() else \"CPU\"}')" 2>/dev/null || {
    echo "错误: 无法检查GPU状态，请检查PyTorch安装"
    exit 1
}

# 启动后台训练
echo "=========================================="
echo "启动SSC ResNet50训练..."
echo "日志文件: $LOG_FILE"
echo "开始时间: $(date)"
echo "=========================================="

# 使用nohup在后台运行训练脚本
nohup python ssc_train.py > "$LOG_FILE" 2>&1 &
TRAIN_PID=$!

# 保存进程ID
echo "$TRAIN_PID" > "$PID_FILE"

echo "训练已启动，进程ID: $TRAIN_PID"
echo ""
echo "使用以下命令管理训练:"
echo "  查看实时日志: tail -f $LOG_FILE"
echo "  查看进程状态: ps -p $TRAIN_PID"
echo "  停止训练: kill $TRAIN_PID && rm $PID_FILE"
echo ""

# 等待几秒检查进程是否成功启动
sleep 3
if ps -p "$TRAIN_PID" > /dev/null 2>&1; then
    echo "✓ 训练进程启动成功！"
    echo "✓ 可以安全断开SSH连接，训练将继续在后台运行"
    echo ""
    echo "提示: 使用 'screen -r' 或查看日志文件监控训练进度"
else
    echo "✗ 警告: 训练进程可能启动失败，请检查日志文件: $LOG_FILE"
    rm -f "$PID_FILE"
    exit 1
fi

