#!/usr/bin/env bash
#
# ST-SACLF AdaIN (train.py) 五数据集批量训练
# 数据集: Painting91, Pandora, ArtBench, FashionStyle14, Arch
# 默认参数沿用 train.py，每库 runs=3，四项指标
# 结果合并写入 ieee_access_paperdata/ST-SACLF_multiple.md（格式对齐 BarlowTwins_multiple.md）
#
# 用法（均在项目根目录执行）::
#   ./ST-SACLF-ncc_main/pytorch-AdaIN/run_st_saclf_train_bat.sh              # 后台 + nohup
#   ./ST-SACLF-ncc_main/pytorch-AdaIN/run_st_saclf_train_bat.sh fg           # 前台（调试）
#
# 管理: ./ST-SACLF-ncc_main/pytorch-AdaIN/manage_st_saclf_train_bat.sh {start|stop|status|tail|partial|result|…}

set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SELF_DIR/../.." && pwd)"
cd "$ROOT"

PID_FILE="$SELF_DIR/st_saclf_bat.pid"
LASTLOG_FILE="$SELF_DIR/st_saclf_bat.lastlog"
PARTIAL_INFO_FILE="$SELF_DIR/st_saclf_bat.partialdir"
LOG_DIR="$SELF_DIR/logs"
RESULT_MD="$ROOT/ieee_access_paperdata/ST-SACLF_multiple.md"
DATA_BASE="/mnt/codes/data/style"
RUNS=3
MAX_ITER=10000

# label|num_classes|相对 data_base 的子目录（ArtBench 对应 Artbench）
DATASETS=(
  "Painting91|13|Painting91"
  "Pandora|12|Pandora"
  "ArtBench|10|Artbench"
  "FashionStyle14|14|FashionStyle14"
  "Arch|25|Arch"
)

TIMESTAMP="${ST_SACLF_BAT_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${ST_SACLF_BAT_LOG:-$LOG_DIR/st_saclf_bat_${TIMESTAMP}.log}"
PARTIAL_DIR="${ST_SACLF_BAT_PARTIAL:-$LOG_DIR/st_saclf_partials_${TIMESTAMP}}"

mkdir -p "$LOG_DIR" "$PARTIAL_DIR"
mkdir -p "$ROOT/ieee_access_paperdata"

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
  PARTIAL_DIR="$PARTIAL_DIR" RESULT_MD="$RESULT_MD" RUNS="$RUNS" DATA_BASE="$DATA_BASE" \
  "$PYTHON_BIN" - <<'PY'
import glob
import os
import re
from datetime import datetime

partial_dir = os.environ["PARTIAL_DIR"]
result_md = os.environ["RESULT_MD"]
runs = int(os.environ["RUNS"])
data_base = os.environ["DATA_BASE"]
if not data_base.endswith("/"):
    data_base += "/"

dataset_order = ["Painting91", "Pandora", "ArtBench", "FashionStyle14", "Arch"]
dataset_rel = {
    "Painting91": "Painting91",
    "Pandora": "Pandora",
    "ArtBench": "Artbench",
    "FashionStyle14": "FashionStyle14",
    "Arch": "Arch",
}
metric_sections = [
    "Accuracy",
    "Macro-F1",
    "Weighted-F1",
    "Balanced Accuracy",
]

partials = {}
for path in sorted(glob.glob(os.path.join(partial_dir, "*.md"))):
    name = os.path.splitext(os.path.basename(path))[0]
    with open(path, encoding="utf-8") as f:
        partials[name] = f.read()

max_iter = clf_epochs = "?"
if partials:
    m = re.search(r"max_iter=(\d+),\s*clf_epochs=(\d+)", next(iter(partials.values())))
    if m:
        max_iter, clf_epochs = m.group(1), m.group(2)

def extract_table_row(text: str, section: str) -> str:
    pat = rf"### {re.escape(section)}\s*\n\n(\|.+\|\n\|[-| ]+\|\n)(\|.+\|)"
    hit = re.search(pat, text)
    if not hit:
        return ""
    return hit.group(2).strip()

def extract_summary_row(text: str) -> str:
    pat = r"## 汇总总表\s*\n\n(\|.+\|\n\|[-| ]+\|\n)(\|.+\|)"
    hit = re.search(pat, text)
    if not hit:
        return ""
    return hit.group(2).strip()

run_headers = [f"run{i}" for i in range(1, runs + 1)]
lines = [
    "# ST-SACLF (AdaIN) 多数据集多次实验",
    "",
    f"## ST-SACLF (AdaIN) benchmark ({', '.join(dataset_order)}) "
    f"(max_iter={max_iter}, clf_epochs={clf_epochs}, runs={runs}) — "
    f"{datetime.now():%Y-%m-%d %H:%M:%S}",
    "",
    f"_data_base=`{data_base}`_",
    "",
    f"_命令: `./ST-SACLF-ncc_main/pytorch-AdaIN/run_st_saclf_train_bat.sh` → "
    f"`train.py` × {len(dataset_order)} 数据集_",
    "",
]

for section in metric_sections:
    lines += [
        f"### {section}",
        "",
        "| Dataset | num_classes | "
        + " | ".join(run_headers)
        + " | mean±std | data_root |",
        "|" + "|".join(["---------"] * (4 + runs)) + "|",
    ]
    for ds in dataset_order:
        row = extract_table_row(partials.get(ds, ""), section)
        if row:
            lines.append(row)
        else:
            failed = " | ".join(["FAILED"] * runs)
            lines.append(
                f"| {ds} | ? | {failed} | FAILED | `{data_base}{dataset_rel.get(ds, ds)}` |"
            )
    lines.append("")

lines += [
    "## 汇总总表",
    "",
    "| Dataset | num_classes | Accuracy | Macro-F1 | Weighted-F1 | Balanced Accuracy |",
    "|---------|-------------|---------|---------|---------|---------|",
]
for ds in dataset_order:
    row = extract_summary_row(partials.get(ds, ""))
    if row:
        lines.append(row)
    else:
        lines.append(f"| {ds} | ? | FAILED | FAILED | FAILED | FAILED |")
lines.append("")

os.makedirs(os.path.dirname(os.path.abspath(result_md)), exist_ok=True)
with open(result_md, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"合并完成: {result_md}")
PY
}

run_batch() {
  echo "ROOT=$ROOT"
  echo "训练脚本: $SELF_DIR/train.py"
  echo "日志: $LOG_FILE"
  echo "部分结果: $PARTIAL_DIR"
  echo "最终结果: $RESULT_MD"
  echo "runs=$RUNS, max_iter=$MAX_ITER, 数据集数=${#DATASETS[@]}"
  echo "GPU 检查:"
  "$PYTHON_BIN" -c "import torch; print('  CUDA:', torch.cuda.is_available(), 'count:', torch.cuda.device_count())" 2>/dev/null || true
  echo ""

  local idx=0 total="${#DATASETS[@]}"
  for entry in "${DATASETS[@]}"; do
    idx=$((idx + 1))
    IFS='|' read -r label n_cls rel <<<"$entry"
    data_root="${DATA_BASE%/}/${rel}/"
    partial_md="${PARTIAL_DIR}/${label}.md"

    echo "========== [$idx/$total] $label (classes=$n_cls) =========="
    echo "  data_root=$data_root"
    echo "  partial_md=$partial_md"

    "$PYTHON_BIN" "$SELF_DIR/train.py" \
      --data_root "$data_root" \
      --num_classes "$n_cls" \
      --runs "$RUNS" \
      --max_iter "$MAX_ITER" \
      --result_md "$partial_md" \
      --merge_result_md "$RESULT_MD" \
      --partial_dir "$PARTIAL_DIR" \
      --dataset_label "$label"
  done

  echo ""
  echo "========== 最终合并 Markdown =========="
  merge_partial_markdown
}

run_foreground() {
  echo "$LOG_FILE" >"$LASTLOG_FILE"
  echo "$PARTIAL_DIR" >"$PARTIAL_INFO_FILE"
  echo "前台运行（输出同时写入日志）..."
  run_batch 2>&1 | tee -a "$LOG_FILE"
}

run_background() {
  if [[ -f "$PID_FILE" ]]; then
    OLD_PID="$(cat "$PID_FILE")"
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
      echo "已有 ST-SACLF 批量任务在运行 (PID=$OLD_PID)。"
      exit 1
    fi
    rm -f "$PID_FILE"
  fi

  echo "$LOG_FILE" >"$LASTLOG_FILE"
  echo "$PARTIAL_DIR" >"$PARTIAL_INFO_FILE"
  echo "后台运行: nohup → $LOG_FILE"
  nohup env \
    ST_SACLF_BAT_LOG="$LOG_FILE" \
    ST_SACLF_BAT_PARTIAL="$PARTIAL_DIR" \
    ST_SACLF_BAT_TIMESTAMP="$TIMESTAMP" \
    "$SELF_DIR/run_st_saclf_train_bat.sh" _batch >>"$LOG_FILE" 2>&1 &
  echo $! >"$PID_FILE"
  echo "PID=$(cat "$PID_FILE") 已写入 $PID_FILE"
  echo "查看日志: ./ST-SACLF-ncc_main/pytorch-AdaIN/manage_st_saclf_train_bat.sh tail"
  echo "中间结果: ./ST-SACLF-ncc_main/pytorch-AdaIN/manage_st_saclf_train_bat.sh partial"
}

run_merge_only() {
  if [[ -n "${ST_SACLF_BAT_PARTIAL:-}" ]]; then
    PARTIAL_DIR="$ST_SACLF_BAT_PARTIAL"
  elif [[ -f "$PARTIAL_INFO_FILE" ]]; then
    PARTIAL_DIR="$(cat "$PARTIAL_INFO_FILE")"
  else
    PARTIAL_DIR="$(ls -td "$LOG_DIR"/st_saclf_partials_* 2>/dev/null | head -1 || true)"
  fi
  if [[ -z "${PARTIAL_DIR:-}" || ! -d "$PARTIAL_DIR" ]]; then
    echo "错误: 未找到 partial 目录"
    exit 1
  fi
  echo "合并 partial: $PARTIAL_DIR → $RESULT_MD"
  merge_partial_markdown
}

case "${1:-}" in
  fg|foreground|front) run_foreground ;;
  merge) run_merge_only ;;
  _batch)
    LOG_FILE="${ST_SACLF_BAT_LOG:-$LOG_FILE}"
    PARTIAL_DIR="${ST_SACLF_BAT_PARTIAL:-$PARTIAL_DIR}"
    echo "$PARTIAL_DIR" >"$PARTIAL_INFO_FILE"
    run_batch
    ;;
  *) run_background ;;
esac
