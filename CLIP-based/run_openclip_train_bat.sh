#!/usr/bin/env bash
#
# OpenCLIP Community Variants (openclip_community_variants_train.py) 五数据集批量评测
# 本地权重：linear_probe=vit_large_patch16_224.pth，zero_shot=ViT-L-14-openai.pt
# 结果合并写入 ieee_access_paperdata/clip-based_multiple.md
#
# 用法（均在项目根目录执行）::
#   ./CLIP-based/run_openclip_train_bat.sh              # 后台 + nohup
#   ./CLIP-based/run_openclip_train_bat.sh fg           # 前台（调试）
#
# 管理: ./CLIP-based/manage_openclip_train_bat.sh {start|stop|status|tail|result|…}

set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SELF_DIR/.." && pwd)"
cd "$ROOT"

PID_FILE="$SELF_DIR/openclip_bat.pid"
LASTLOG_FILE="$SELF_DIR/openclip_bat.lastlog"
LOG_DIR="$SELF_DIR/logs"
RESULT_MD="$ROOT/ieee_access_paperdata/clip-based_multiple.md"
DATA_BASE="/mnt/codes/data/style"
RUNS=3
MODES=(zero_shot linear_probe)

# label|num_classes|相对 data_base 的子目录（ArtBench 对应 Artbench）
DATASETS=(
  "Painting91|13|Painting91"
  "Pandora|12|Pandora"
  "ArtBench|10|Artbench"
  "FashionStyle14|14|FashionStyle14"
  "Arch|25|Arch"
)

TIMESTAMP="${OPENCLIP_BAT_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${OPENCLIP_BAT_LOG:-$LOG_DIR/openclip_bat_${TIMESTAMP}.log}"
PARTIAL_DIR="${OPENCLIP_BAT_PARTIAL:-$LOG_DIR/openclip_partials_${TIMESTAMP}}"

mkdir -p "$LOG_DIR" "$PARTIAL_DIR"
mkdir -p "$ROOT/model" "$ROOT/log" "$ROOT/pretrainModels" "$ROOT/ieee_access_paperdata"

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

merge_partial_markdown() {
  ROOT="$ROOT" PARTIAL_DIR="$PARTIAL_DIR" RESULT_MD="$RESULT_MD" RUNS="$RUNS" DATA_BASE="$DATA_BASE" \
  "$PYTHON_BIN" - <<'PY'
import os
import sys

sys.path.insert(0, os.path.join(os.environ["ROOT"], "CLIP-based"))
from openclip_community_variants_train import merge_all_modes_to_file

merge_all_modes_to_file(
    os.environ["PARTIAL_DIR"],
    os.environ["RESULT_MD"],
    int(os.environ["RUNS"]),
    os.environ["DATA_BASE"],
)
print(f"合并完成: {os.environ['RESULT_MD']}")
PY
}

run_batch() {
  echo "ROOT=$ROOT"
  echo "训练脚本: $ROOT/CLIP-based/openclip_community_variants_train.py"
  echo "日志: $LOG_FILE"
  echo "部分结果: $PARTIAL_DIR"
  echo "最终结果: $RESULT_MD"
  echo "modes=${MODES[*]}, runs=$RUNS, 数据集数=${#DATASETS[@]}"
  echo "GPU 检查:"
  "$PYTHON_BIN" -c "import torch; print('  CUDA:', torch.cuda.is_available(), 'count:', torch.cuda.device_count())" 2>/dev/null || true
  echo ""

  for mode in "${MODES[@]}"; do
    echo "########## 模式: $mode ##########"
    local idx=0 total="${#DATASETS[@]}"
    for entry in "${DATASETS[@]}"; do
      idx=$((idx + 1))
      IFS='|' read -r label n_cls rel <<<"$entry"
      data_root="${DATA_BASE%/}/${rel}/"
      partial_md="${PARTIAL_DIR}/${label}_${mode}.md"

      echo "========== [$mode][$idx/$total] $label (classes=$n_cls) =========="
      echo "  data_root=$data_root"
      echo "  partial_md=$partial_md"

      "$PYTHON_BIN" "$ROOT/CLIP-based/openclip_community_variants_train.py" \
        --data_root "$data_root" \
        --num_classes "$n_cls" \
        --mode "$mode" \
        --runs "$RUNS" \
        --result_md "$partial_md" \
        --dataset_label "$label"

      ROOT="$ROOT" PARTIAL_DIR="$PARTIAL_DIR" RESULT_MD="$RESULT_MD" RUNS="$RUNS" \
        DATA_BASE="$DATA_BASE" "$PYTHON_BIN" - <<'PY'
import os, sys
sys.path.insert(0, os.path.join(os.environ["ROOT"], "CLIP-based"))
from openclip_community_variants_train import merge_all_modes_to_file
merge_all_modes_to_file(
    os.environ["PARTIAL_DIR"],
    os.environ["RESULT_MD"],
    int(os.environ["RUNS"]),
    os.environ["DATA_BASE"],
)
print(f"总表已更新: {os.environ['RESULT_MD']}")
PY
    done
  done

  echo ""
  echo "========== 最终合并 Markdown =========="
  merge_partial_markdown
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
      echo "已有 OpenCLIP 批量任务在运行 (PID=$OLD_PID)。请先: ./CLIP-based/manage_openclip_train_bat.sh stop"
      exit 1
    fi
    rm -f "$PID_FILE"
  fi

  echo "$LOG_FILE" >"$LASTLOG_FILE"
  echo "后台运行: nohup → $LOG_FILE"
  nohup env \
    OPENCLIP_BAT_LOG="$LOG_FILE" \
    OPENCLIP_BAT_PARTIAL="$PARTIAL_DIR" \
    OPENCLIP_BAT_TIMESTAMP="$TIMESTAMP" \
    ROOT="$ROOT" \
    "$SELF_DIR/run_openclip_train_bat.sh" _batch >>"$LOG_FILE" 2>&1 &
  echo $! >"$PID_FILE"
  echo "PID=$(cat "$PID_FILE") 已写入 $PID_FILE"
  echo "查看日志: ./CLIP-based/manage_openclip_train_bat.sh tail"
}

case "${1:-}" in
  fg|foreground|front) run_foreground ;;
  _batch)
    LOG_FILE="${OPENCLIP_BAT_LOG:-$LOG_FILE}"
    PARTIAL_DIR="${OPENCLIP_BAT_PARTIAL:-$PARTIAL_DIR}"
    run_batch
    ;;
  *) run_background ;;
esac
