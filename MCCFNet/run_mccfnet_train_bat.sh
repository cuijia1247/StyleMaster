#!/usr/bin/env bash
#
# MCCFNet 六数据集批量训练（与 barlowtwins/simclr 六库一致；默认 epochs=3；每库 5 次，结果含各次准确率与 mean±std，写入 MCCFNet/mccfnet_result.md）
# 适合 SSH 离线：默认 nohup 后台。
#
# 用法（均在项目根目录执行）::
#   ./MCCFNet/run_mccfnet_train_bat.sh              # 后台 + nohup
#   ./MCCFNet/run_mccfnet_train_bat.sh fg         # 前台
#
# 依赖：conda 环境 ssc。

set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SELF_DIR/.." && pwd)"
cd "$ROOT"

PID_FILE="$SELF_DIR/mccfnet_train_bat.pid"
LOG_DIR="$SELF_DIR/logs"
mkdir -p "$LOG_DIR"
mkdir -p "$ROOT/pretrainModels/hub" "$ROOT/model" "$ROOT/log"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/mccfnet_bat_${TIMESTAMP}.log"
RESULT_MD="$SELF_DIR/mccfnet_result.md"

export TORCH_HOME="${TORCH_HOME:-$ROOT/pretrainModels}"

if command -v conda &>/dev/null; then
  # shellcheck source=/dev/null
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate ssc 2>/dev/null || true
fi

PYTHON_BIN="python3"
if command -v python &>/dev/null && python -c "import torch" 2>/dev/null; then
  PYTHON_BIN="python"
elif ! "$PYTHON_BIN" -c "import torch" 2>/dev/null; then
  echo "错误: 未找到带 PyTorch 的 python，请先激活环境（如 conda activate ssc）"
  exit 1
fi

CMD=(
  "$PYTHON_BIN" "$ROOT/MCCFNet/mccfnet_train.py"
  --benchmark_all
  --benchmark_runs 5
  --epochs 3
  --data_base "/mnt/codes/data/style/"
  --result_md "$RESULT_MD"
)

echo "ROOT=$ROOT"
echo "日志: $LOG_FILE"
echo "结果: $RESULT_MD"
echo "GPU:"
"$PYTHON_BIN" -c "import torch; print('  CUDA:', torch.cuda.is_available(), 'count:', torch.cuda.device_count())" 2>/dev/null || true
echo ""

run_foreground() {
  echo "前台运行..."
  "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
}

run_background() {
  if [[ -f "$PID_FILE" ]]; then
    OLD_PID="$(cat "$PID_FILE")"
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
      echo "已有 MCCFNet 批量任务在运行 (PID=$OLD_PID)。请先: ./MCCFNet/manage_mccfnet_train_bat.sh stop"
      exit 1
    fi
    rm -f "$PID_FILE"
  fi

  echo "后台: nohup → $LOG_FILE"
  nohup "${CMD[@]}" >>"$LOG_FILE" 2>&1 &
  echo $! >"$PID_FILE"
  echo "PID=$(cat "$PID_FILE") → $PID_FILE"
  echo "查看日志: ./MCCFNet/manage_mccfnet_train_bat.sh tail"
}

case "${1:-}" in
  fg|foreground|front) run_foreground ;;
  *)                   run_background ;;
esac
