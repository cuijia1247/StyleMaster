#!/usr/bin/env bash
#
# Ours SSC-ResNet50 (ieee_ssc_train_resnet.py) 四数据集批量训练
# 数据集: Pandora, ArtBench, FashionStyle14, Arch（跳过 Painting91）
# 默认参数沿用 ssc_train_resnet_copy.py（parameter_load），每库 runs=3，四项指标
# 结果追加写入 ieee_access_paperdata/ours_multiple.md（保留已有 Painting91 等记录）
#
# 用法（均在项目根目录执行）::
#   ./ieee_access_codes/run_ieee_ssc_train_bat.sh              # 后台 + nohup
#   ./ieee_access_codes/run_ieee_ssc_train_bat.sh fg           # 前台（调试）
#
# 管理: ./ieee_access_codes/manage_ieee_ssc_train_bat.sh {start|stop|status|tail|result|…}

set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SELF_DIR/.." && pwd)"
cd "$ROOT"

PID_FILE="$SELF_DIR/ieee_ssc_bat.pid"
LASTLOG_FILE="$SELF_DIR/ieee_ssc_bat.lastlog"
LOG_DIR="$SELF_DIR/logs"
RESULT_MD="$ROOT/ieee_access_paperdata/ours_multiple.md"
DATA_BASE="/mnt/codes/data/style"
RUNS=3
PRE_FEATURE_PATH="${PRE_FEATURE_PATH:-$ROOT/pretrainFeatures}"
MODEL_PATH="${MODEL_PATH:-$ROOT/model}"

# label|num_classes|相对 data_base 的子目录（ArtBench 对应 Artbench）
DATASETS=(
  "Pandora|12|Pandora"
  "ArtBench|10|Artbench"
  "FashionStyle14|14|FashionStyle14"
  "Arch|25|Arch"
)

TIMESTAMP="${IEEE_SSC_BAT_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${IEEE_SSC_BAT_LOG:-$LOG_DIR/ieee_ssc_bat_${TIMESTAMP}.log}"

mkdir -p "$LOG_DIR"
mkdir -p "$ROOT/model" "$ROOT/log" "$ROOT/pretrainFeatures" "$ROOT/ieee_access_paperdata"

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

run_batch() {
  echo "ROOT=$ROOT"
  echo "训练脚本: $ROOT/ieee_ssc_train_resnet.py"
  echo "日志: $LOG_FILE"
  echo "最终结果（追加）: $RESULT_MD"
  echo "data_base=$DATA_BASE, runs=$RUNS, 数据集数=${#DATASETS[@]}（跳过 Painting91）"
  echo "pre_feature_path=$PRE_FEATURE_PATH"
  echo "GPU 检查:"
  "$PYTHON_BIN" -c "import torch; print('  CUDA:', torch.cuda.is_available(), 'count:', torch.cuda.device_count())" 2>/dev/null || true
  echo ""

  local idx=0 total="${#DATASETS[@]}"
  for entry in "${DATASETS[@]}"; do
    idx=$((idx + 1))
    IFS='|' read -r label n_cls rel <<<"$entry"
    data_root="${DATA_BASE%/}/${rel}/"

    echo "========== [$idx/$total] $label (classes=$n_cls) =========="
    echo "  data_root=$data_root"

    "$PYTHON_BIN" "$ROOT/ieee_ssc_train_resnet.py" \
      --dataset_name "$rel" \
      --data_base "$DATA_BASE" \
      --runs "$RUNS" \
      --result_md "$RESULT_MD" \
      --append_result \
      --pre_feature_path "$PRE_FEATURE_PATH" \
      --model_path "$MODEL_PATH"
  done

  echo ""
  echo "========== 训练完成 =========="
  echo "结果已追加: $RESULT_MD"
}

run_foreground() {
  echo "$LOG_FILE" >"$LASTLOG_FILE"
  echo "前台运行（输出同时写入日志）..."
  run_batch 2>&1 | tee -a "$LOG_FILE"
}

run_background() {
  if [[ -f "$PID_FILE" ]]; then
    OLD_PID="$(cat "$PID_FILE")"
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
      echo "已有 SSC-ResNet50 批量任务在运行 (PID=$OLD_PID)。"
      echo "请先: ./ieee_access_codes/manage_ieee_ssc_train_bat.sh stop"
      exit 1
    fi
    rm -f "$PID_FILE"
  fi

  echo "$LOG_FILE" >"$LASTLOG_FILE"
  echo "后台运行: nohup → $LOG_FILE"
  nohup env \
    IEEE_SSC_BAT_LOG="$LOG_FILE" \
    IEEE_SSC_BAT_TIMESTAMP="$TIMESTAMP" \
    PRE_FEATURE_PATH="$PRE_FEATURE_PATH" \
    MODEL_PATH="$MODEL_PATH" \
    "$SELF_DIR/run_ieee_ssc_train_bat.sh" _batch >>"$LOG_FILE" 2>&1 &
  echo $! >"$PID_FILE"
  echo "PID=$(cat "$PID_FILE") 已写入 $PID_FILE"
  echo "查看日志: ./ieee_access_codes/manage_ieee_ssc_train_bat.sh tail"
}

case "${1:-}" in
  fg|foreground|front) run_foreground ;;
  _batch)
    LOG_FILE="${IEEE_SSC_BAT_LOG:-$LOG_FILE}"
    run_batch
    ;;
  *) run_background ;;
esac
