#!/bin/bash

# ResNet50 超参数网格搜索后台运行脚本
# Author: cuijia1247
# Date: 2026-03-29
# 功能：使用 nohup 在后台运行参数优化，断开 SSH 后仍可继续
# 用法：从项目根目录执行 ./remote_sh/run_resnet50_param_optim.sh

# 获取项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

# 创建结果与日志目录
mkdir -p param_optim/resnet50/logs

# 生成时间戳
TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S")
NOHUP_LOG="param_optim/resnet50/nohup_${TIMESTAMP}.log"
PID_FILE="resnet50_param_optim.pid"

# 检查是否已有搜索进程在运行
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "参数优化进程已在运行 (PID: $OLD_PID)"
        echo "如需重新启动，请先执行: kill $OLD_PID && rm $PID_FILE"
        exit 1
    else
        echo "清理旧的 PID 文件"
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

if [ "$CONDA_DEFAULT_ENV" != "ssc" ]; then
    echo "错误: 无法激活 conda 环境 'ssc'"
    echo "请确保环境存在: conda env list"
    exit 1
fi

# 检查 GPU 可用性
echo "检查 GPU 状态..."
python -c "import torch; print(f'CUDA 可用: {torch.cuda.is_available()}'); print(f'CUDA 设备数量: {torch.cuda.device_count()}')" 2>/dev/null || {
    echo "警告: 无法检查 GPU 状态"
}

# 启动后台参数搜索
echo "=========================================="
echo "启动 ResNet50 超参数网格搜索..."
echo "nohup 日志: $NOHUP_LOG"
echo "各组合详细日志: param_optim/resnet50/logs/"
echo "汇总结果 CSV: param_optim/resnet50/grid_search_results.csv"
echo "开始时间: $(date)"
echo "=========================================="

nohup python param_optim/resnet50_param_optim.py > "$NOHUP_LOG" 2>&1 &
OPTIM_PID=$!

echo "$OPTIM_PID" > "$PID_FILE"

echo "参数优化已启动，进程 ID: $OPTIM_PID"
echo ""
echo "使用以下命令管理进程:"
echo "  查看实时输出: tail -f $NOHUP_LOG"
echo "  查看进程状态: ps -p $OPTIM_PID"
echo "  查看汇总结果: cat param_optim/resnet50/grid_search_results.csv"
echo "  停止搜索:     kill $OPTIM_PID && rm $PID_FILE"
echo ""

# 等待几秒确认进程成功启动
sleep 3
if ps -p "$OPTIM_PID" > /dev/null 2>&1; then
    echo "✓ 参数优化进程启动成功！"
    echo "✓ 可以安全断开 SSH 连接，搜索将继续在后台运行"
else
    echo "✗ 警告: 进程可能启动失败，请检查日志: $NOHUP_LOG"
    rm -f "$PID_FILE"
    exit 1
fi
