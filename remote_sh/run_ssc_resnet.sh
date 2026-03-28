#!/bin/bash

# SSC ResNet 训练后台运行脚本
# Author: cuijia1247
# Date: 2026-03-28
# 功能：使用 nohup 在后台运行训练，断开 SSH 后仍可继续
# 用法：从项目根目录执行 ./remote_sh/run_ssc_resnet.sh

# 获取脚本所在目录的父目录（项目根目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
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
        echo "训练进程已在运行 (PID: $OLD_PID)"
        echo "如需重新启动，请先运行: ./remote_sh/manage_ssc_resnet.sh stop"
        exit 1
    else
        echo "清理旧的PID文件"
        rm -f "$PID_FILE"
    fi
fi

# 检查 conda 环境
if ! command -v conda &> /dev/null; then
    echo "错误: conda 未安装或未在 PATH 中"
    exit 1
fi

# 激活 conda 环境
echo "激活 conda 环境: ssc"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ssc

# 检查环境是否正确激活
if [ "$CONDA_DEFAULT_ENV" != "ssc" ]; then
    echo "错误: 无法激活 conda 环境 'ssc'"
    echo "请确保环境存在: conda env list"
    exit 1
fi

# 检查 GPU 可用性
echo "检查 GPU 状态..."
python -c "import torch; print(f'CUDA 可用: {torch.cuda.is_available()}'); print(f'CUDA 设备数量: {torch.cuda.device_count()}'); print(f'当前设备: {torch.cuda.current_device() if torch.cuda.is_available() else \"CPU\"}')" 2>/dev/null || {
    echo "警告: 无法检查 GPU 状态"
}

# 启动后台训练
echo "=========================================="
echo "启动 SSC ResNet50 训练..."
echo "日志文件: $LOG_FILE"
echo "开始时间: $(date)"
echo "=========================================="

# 使用 nohup 在后台运行训练脚本
nohup python ssc_train_resnet.py > "$LOG_FILE" 2>&1 &
TRAIN_PID=$!

# 保存进程 ID
echo "$TRAIN_PID" > "$PID_FILE"

echo "训练已启动，进程 ID: $TRAIN_PID"
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
    echo "✓ 可以安全断开 SSH 连接，训练将继续在后台运行"
else
    echo "✗ 警告: 训练进程可能启动失败，请检查日志文件: $LOG_FILE"
    rm -f "$PID_FILE"
    exit 1
fi
