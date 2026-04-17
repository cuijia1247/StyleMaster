#!/usr/bin/env bash
#
# SimCLR 六数据集批量训练（每库跑 3 遍，mean±std 写入 selfsupervised/simclr_result.md）
# 适合 SSH 离线：默认 nohup 后台，断开终端仍继续。
#
# 用法（均在项目根目录执行）::
#   ./selfsupervised/run_simclr_train_bat.sh              # 后台 + nohup
#   ./selfsupervised/run_simclr_train_bat.sh fg           # 前台（调试，Ctrl+C 结束）
#
# 依赖：conda 环境 ssc（与项目其它脚本一致）；若无 conda 则使用当前 python3。

set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SELF_DIR/.." && pwd)"
cd "$ROOT"

PID_FILE="$SELF_DIR/simclr_bat.pid"
LOG_DIR="$SELF_DIR/logs"
mkdir -p "$LOG_DIR"
mkdir -p "$ROOT/pretrainModels" "$ROOT/model" "$ROOT/log"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/simclr_bat_${TIMESTAMP}.log"
RESULT_MD="$SELF_DIR/simclr_result.md"

export TORCH_HOME="${TORCH_HOME:-$ROOT/pretrainModels}"

# 可选 conda
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
  "$PYTHON_BIN" "$ROOT/selfsupervised/simclr_train.py"
  --benchmark_all
  --runs 3
  --result_md "$RESULT_MD"
)

echo "ROOT=$ROOT"
echo "日志: $LOG_FILE"
echo "结果: $RESULT_MD"
echo "GPU 检查:"
"$PYTHON_BIN" -c "import torch; print('  CUDA:', torch.cuda.is_available(), 'count:', torch.cuda.device_count())" 2>/dev/null || true
echo ""

run_foreground() {
  echo "前台运行（输出同时写入日志）..."
  "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
}

run_background() {
  if [[ -f "$PID_FILE" ]]; then
    OLD_PID="$(cat "$PID_FILE")"
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
      echo "已有 SimCLR 批量任务在运行 (PID=$OLD_PID)。请先: ./selfsupervised/manage_simclr_train_bat.sh stop"
      exit 1
    fi
    rm -f "$PID_FILE"
  fi

  echo "后台运行: nohup → $LOG_FILE"
  nohup "${CMD[@]}" >>"$LOG_FILE" 2>&1 &
  echo $! >"$PID_FILE"
  echo "PID=$(cat "$PID_FILE") 已写入 $PID_FILE"
  echo "查看日志: ./selfsupervised/manage_simclr_train_bat.sh tail"
}

case "${1:-}" in
  fg|foreground|front) run_foreground ;;
  *)                   run_background ;;
esac
