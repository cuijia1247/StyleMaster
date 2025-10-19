#!/bin/bash

# SSH 后台运行 SSC Transformer 训练脚本
# Author: cuijia1247
# Date: 2025-01-27
# 支持本地模型加载，完全离线运行

# 设置脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 创建日志目录
mkdir -p log

# 生成时间戳
TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S")
LOG_FILE="log/ssc_background_${TIMESTAMP}.log"
PID_FILE="ssc_train.pid"

# 检查是否已有训练进程在运行
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "训练进程已在运行 (PID: $OLD_PID)"
        echo "如需重新启动，请先运行: ./manage_ssc.sh stop"
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

# 检查本地模型文件
MODEL_FILE="pretrainModels/swin_base_patch4_window7_224.pth"
if [ ! -f "$MODEL_FILE" ]; then
    echo "错误: 本地模型文件不存在: $MODEL_FILE"
    echo "请确保模型文件已正确放置"
    exit 1
fi

# 检查GPU可用性
echo "检查GPU状态..."
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'CUDA设备数量: {torch.cuda.device_count()}')" 2>/dev/null || {
    echo "错误: 无法检查GPU状态，请检查PyTorch安装"
    exit 1
}

# 启动后台训练
echo "启动SSC Transformer训练..."
echo "日志文件: $LOG_FILE"
echo "开始时间: $(date)"

# 使用nohup在后台运行训练脚本
nohup python ssc_train_transformer.py > "$LOG_FILE" 2>&1 &
TRAIN_PID=$!

# 保存进程ID
echo "$TRAIN_PID" > "$PID_FILE"

echo "训练已启动，进程ID: $TRAIN_PID"
echo "使用以下命令管理训练:"
echo "  查看状态: ./manage_ssc.sh status"
echo "  查看日志: ./manage_ssc.sh tail"
echo "  停止训练: ./manage_ssc.sh stop"

# 等待几秒检查进程是否成功启动
sleep 3
if ps -p "$TRAIN_PID" > /dev/null 2>&1; then
    echo "训练进程启动成功！"
    echo "可以安全断开SSH连接，训练将继续在后台运行"
else
    echo "警告: 训练进程可能启动失败，请检查日志文件: $LOG_FILE"
    rm -f "$PID_FILE"
    exit 1
fi
