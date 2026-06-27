#!/usr/bin/env bash
#
# VICReg Pandora 数据集后台训练（nohup，SSH 断开仍继续）
# 默认: Pandora, parameter_load() 超参, runs=3, 结果合并至 vicreg_multiple.md
#
# 用法（均在项目根目录执行）::
#   ./selfsupervised/run_vicreg_train.sh              # 后台 + nohup
#   ./selfsupervised/run_vicreg_train.sh fg           # 前台（调试）
#
# 管理: ./selfsupervised/manage_vicreg_train.sh {status|tail|stop|progress|result}

set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SELF_DIR/.." && pwd)"
cd "$ROOT"

PID_FILE="$SELF_DIR/vicreg_train.pid"
LASTLOG_FILE="$SELF_DIR/vicreg_train.lastlog"
LOG_DIR="$SELF_DIR/logs"
RESULT_MD="$ROOT/ieee_access_paperdata/vicreg_multiple.md"
DATA_ROOT="/mnt/codes/data/style/Pandora"

mkdir -p "$LOG_DIR" "$ROOT/model" "$ROOT/log" "$ROOT/pretrainModels" "$ROOT/ieee_access_paperdata"

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

MODE="bg"
for arg in "$@"; do
  case "$arg" in
    fg|foreground|front) MODE="fg" ;;
  esac
done

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/vicreg_pandora_${TIMESTAMP}.log"

CMD=(
  "$PYTHON_BIN" "$ROOT/vicreg_train.py"
  --data_root "$DATA_ROOT"
  --num_classes 12
  --runs 3
  --merge_result
  --result_md "$RESULT_MD"
)

echo "ROOT=$ROOT"
echo "数据集: $DATA_ROOT (Pandora, 12 classes)"
echo "日志: $LOG_FILE"
echo "结果: $RESULT_MD"
echo "命令: ${CMD[*]}"
echo "GPU 检查:"
"$PYTHON_BIN" -c "import torch; print('  CUDA:', torch.cuda.is_available(), 'count:', torch.cuda.device_count())" 2>/dev/null || true
echo ""

run_foreground() {
  echo "$LOG_FILE" >"$LASTLOG_FILE"
  echo "前台运行（输出同时写入日志）..."
  "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
}

run_background() {
  if [[ -f "$PID_FILE" ]]; then
    OLD_PID="$(cat "$PID_FILE")"
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
      echo "已有 VICReg 任务在运行 (PID=$OLD_PID)。请先: ./selfsupervised/manage_vicreg_train.sh stop"
      exit 1
    fi
    rm -f "$PID_FILE"
  fi

  echo "$LOG_FILE" >"$LASTLOG_FILE"
  echo "后台运行: nohup → $LOG_FILE"
  nohup "${CMD[@]}" >>"$LOG_FILE" 2>&1 &
  echo $! >"$PID_FILE"
  echo "PID=$(cat "$PID_FILE") 已写入 $PID_FILE"
  echo "查看进度: ./selfsupervised/manage_vicreg_train.sh status"
  echo "实时日志: ./selfsupervised/manage_vicreg_train.sh tail"
}

case "$MODE" in
  fg) run_foreground ;;
  *)  run_background ;;
esac
