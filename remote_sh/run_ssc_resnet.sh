#!/bin/bash

# SSC ResNet 后台训练：入口 ssc_train_resnet_copy.py
#
# 超参唯一来源：ssc_train_resnet_copy.py 内 parameter_load() 与 parse_train_args；
# 与 param_optim / 网格搜索无关，本脚本也不写入任何网格搜索结果。
# 未设置环境变量时仅执行：python ssc_train_resnet_copy.py（由 Python 内默认值决定）。
#
# 可选：用环境变量覆盖（变量非空时才追加对应 CLI，未设置则完全使用 Python 内默认值）：
#   DATA_ROOT  DATASET_NAME  MODEL_PATH  PRE_FEATURE_PATH  TRAINING_MODE  ITERATIONS
#   EPOCHS  BATCH_SIZE  BASE_LR  IMAGE_SIZE
#   CLASSIFIER_ITERATION  CLASSIFIER_LR  CLASSIFIER_TRAINING_GAP  CLASSIFIER_TEST_GAP
# 示例：DATASET_NAME=WikiArt3 EPOCHS=120 ./remote_sh/run_ssc_resnet.sh
#
# 完整参数说明：python ssc_train_resnet_copy.py -h
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

# 仅当环境变量非空时追加 CLI，否则由 ssc_train_resnet_copy 内 parameter_load / argparse 决定
PY_CMD=(python ssc_train_resnet_copy.py)
_append_if_nonempty() {
    local flag="$1"
    local val="$2"
    [ -n "$val" ] || return 0
    PY_CMD+=("$flag" "$val")
}

_append_if_nonempty --data_root "${DATA_ROOT:-}"
_append_if_nonempty --dataset_name "${DATASET_NAME:-}"
_append_if_nonempty --model_path "${MODEL_PATH:-}"
_append_if_nonempty --pre_feature_path "${PRE_FEATURE_PATH:-}"
_append_if_nonempty --training_mode "${TRAINING_MODE:-}"
_append_if_nonempty --iterations "${ITERATIONS:-}"
_append_if_nonempty --epochs "${EPOCHS:-}"
_append_if_nonempty --batch_size "${BATCH_SIZE:-}"
_append_if_nonempty --base_lr "${BASE_LR:-}"
_append_if_nonempty --image_size "${IMAGE_SIZE:-}"
_append_if_nonempty --classifier_iteration "${CLASSIFIER_ITERATION:-}"
_append_if_nonempty --classifier_lr "${CLASSIFIER_LR:-}"
_append_if_nonempty --classifier_training_gap "${CLASSIFIER_TRAINING_GAP:-}"
_append_if_nonempty --classifier_test_gap "${CLASSIFIER_TEST_GAP:-}"

# 启动后台训练
echo "=========================================="
echo "启动 SSC ResNet: ssc_train_resnet_copy.py（超参仅来自该 Python 脚本；非 param_optim）"
echo "  实际命令: ${PY_CMD[*]}"
echo "日志文件: $LOG_FILE"
echo "开始时间: $(date)"
echo "=========================================="

nohup "${PY_CMD[@]}" > "$LOG_FILE" 2>&1 &
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
