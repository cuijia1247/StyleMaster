#!/usr/bin/env bash
#
# ST-SACLF AdaIN 批量训练：进程 / 日志 / 中间结果管理
# 用法: ./ST-SACLF-ncc_main/pytorch-AdaIN/manage_st_saclf_train_bat.sh {start|stop|status|tail|partial|merge|result|help}

set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SELF_DIR/../.." && pwd)"
cd "$ROOT"

PID_FILE="$SELF_DIR/st_saclf_bat.pid"
LASTLOG_FILE="$SELF_DIR/st_saclf_bat.lastlog"
PARTIAL_INFO_FILE="$SELF_DIR/st_saclf_bat.partialdir"
LOG_DIR="$SELF_DIR/logs"
RESULT_FILE="$ROOT/ieee_access_paperdata/ST-SACLF_multiple.md"

DATASET_ORDER=(Painting91 Pandora ArtBench FashionStyle14 Arch)

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

resolve_partial_dir() {
  if [[ -n "${1:-}" ]]; then
    echo "$1"
    return 0
  fi
  if [[ -f "$PARTIAL_INFO_FILE" ]]; then
    local dir
    dir="$(cat "$PARTIAL_INFO_FILE" 2>/dev/null || true)"
    if [[ -n "$dir" && -d "$dir" ]]; then
      echo "$dir"
      return 0
    fi
  fi
  ls -td "$LOG_DIR"/st_saclf_partials_* 2>/dev/null | head -1 || true
}

resolve_latest_log() {
  local latest=""
  if [[ -f "$LASTLOG_FILE" ]]; then
    latest="$(cat "$LASTLOG_FILE" 2>/dev/null || true)"
  fi
  if [[ -z "${latest:-}" || ! -f "$latest" ]]; then
    latest="$(ls -t "$LOG_DIR"/st_saclf_bat_*.log 2>/dev/null | head -1 || true)"
  fi
  echo "$latest"
}

show_help() {
  echo "ST-SACLF AdaIN 批量训练（train.py × 五数据集 × runs=3，max_iter=10000）"
  echo "结果: ieee_access_paperdata/ST-SACLF_multiple.md"
  echo ""
  echo "用法: $0 {start|stop|restart|status|tail|logs|partial|merge|result|help}"
  echo ""
  echo "  start   - 启动 run_st_saclf_train_bat.sh（nohup 后台）"
  echo "  stop    - 结束批量 shell 及 train.py 子进程"
  echo "  restart - stop 后 start"
  echo "  status  - 进程、GPU、最新日志摘要"
  echo "  tail    - 实时查看最新 batch 日志"
  echo "  logs    - 列出 logs 下 st_saclf_bat 日志与 partial 目录"
  echo "  partial [Dataset] - 查看中间 partial 结果（可指定 Painting91 等）"
  echo "  merge   - 用最新 partial 目录重新合并总表"
  echo "  result  - 打印 ieee_access_paperdata/ST-SACLF_multiple.md（含训练中增量更新）"
  echo ""
  echo "数据集: ${DATASET_ORDER[*]}"
  echo "训练脚本: $SELF_DIR/train.py"
}

start_training() {
  if [[ -f "$PID_FILE" ]]; then
    PID="$(cat "$PID_FILE")"
    if ps -p "$PID" > /dev/null 2>&1; then
      echo -e "${YELLOW}ST-SACLF 批量任务已在运行 (PID: $PID)${NC}"
      return 1
    fi
    rm -f "$PID_FILE"
  fi
  echo -e "${GREEN}启动 ST-SACLF AdaIN 批量训练...${NC}"
  "$SELF_DIR/run_st_saclf_train_bat.sh"
}

stop_training() {
  local stopped=0

  if [[ -f "$PID_FILE" ]]; then
    PID="$(cat "$PID_FILE")"
    if ps -p "$PID" > /dev/null 2>&1; then
      echo -e "${GREEN}停止批量 shell (PID: $PID)...${NC}"
      kill "$PID" 2>/dev/null || true
      for _ in {1..10}; do
        if ! ps -p "$PID" > /dev/null 2>&1; then
          stopped=1
          break
        fi
        sleep 1
      done
      if [[ $stopped -eq 0 ]]; then
        kill -9 "$PID" 2>/dev/null || true
      fi
    fi
    rm -f "$PID_FILE"
  fi

  if pgrep -f "pytorch-AdaIN/train\.py" >/dev/null 2>&1; then
    echo -e "${GREEN}停止 train.py 子进程...${NC}"
    pkill -f "pytorch-AdaIN/train\.py" 2>/dev/null || true
    sleep 2
    pkill -9 -f "pytorch-AdaIN/train\.py" 2>/dev/null || true
    stopped=1
  fi

  if [[ $stopped -eq 0 && ! -f "$PID_FILE" ]]; then
    echo -e "${YELLOW}未找到运行中的 ST-SACLF 批量任务${NC}"
    return 1
  fi
  echo -e "${GREEN}已停止${NC}"
}

restart_training() {
  stop_training || true
  sleep 2
  start_training
}

check_status() {
  local running=0

  if [[ -f "$PID_FILE" ]]; then
    PID="$(cat "$PID_FILE")"
    if ps -p "$PID" > /dev/null 2>&1; then
      running=1
      echo -e "${GREEN}ST-SACLF 批量 shell 运行中${NC}"
      echo "PID: $PID"
      ps -fp "$PID" 2>/dev/null || ps -p "$PID"
      echo ""
    else
      echo -e "${YELLOW}PID 文件存在但 shell 已退出 (PID: $PID)${NC}"
      rm -f "$PID_FILE"
    fi
  fi

  if pgrep -af "pytorch-AdaIN/train\.py" >/dev/null 2>&1; then
    running=1
    echo -e "${GREEN}train.py 进程:${NC}"
    pgrep -af "pytorch-AdaIN/train\.py" || true
    echo ""
  fi

  if [[ $running -eq 0 ]]; then
    echo -e "${RED}ST-SACLF 批量任务未运行${NC}"
  fi

  nvidia-smi 2>/dev/null || echo "nvidia-smi 不可用"
  echo ""

  local partial_dir latest_log
  partial_dir="$(resolve_partial_dir)"
  latest_log="$(resolve_latest_log)"

  if [[ -n "${partial_dir:-}" && -d "$partial_dir" ]]; then
    echo -e "${CYAN}当前 partial 目录: $partial_dir${NC}"
    ls -lh "$partial_dir"/*.md 2>/dev/null || echo "  （尚无 partial md）"
    echo ""
  fi

  if [[ -n "${latest_log:-}" ]]; then
    echo -e "${CYAN}最新日志: $latest_log ($(du -h "$latest_log" | cut -f1))${NC}"
    echo "最后 12 行:"
    tail -12 "$latest_log"
  fi

  if [[ -f "$RESULT_FILE" ]]; then
    echo ""
    echo -e "${CYAN}总表（可能随训练增量更新）: $RESULT_FILE${NC}"
    grep -E '^\| (Painting91|Pandora|ArtBench|FashionStyle14|Arch) ' "$RESULT_FILE" 2>/dev/null | tail -5 || true
  fi

  [[ $running -eq 1 ]]
}

tail_log() {
  local latest
  latest="$(resolve_latest_log)"
  if [[ -z "${latest:-}" ]]; then
    echo -e "${RED}未找到 $LOG_DIR/st_saclf_bat_*.log${NC}"
    return 1
  fi
  echo -e "${GREEN}tail -f $latest${NC} (Ctrl+C 退出)"
  tail -f "$latest"
}

list_logs() {
  echo -e "${GREEN}ST-SACLF 批量日志 ($LOG_DIR):${NC}"
  ls -lht "$LOG_DIR"/st_saclf_bat_*.log 2>/dev/null || echo "暂无"
  echo ""
  echo "部分结果目录:"
  ls -lhtd "$LOG_DIR"/st_saclf_partials_* 2>/dev/null | head -5 || echo "暂无"
  echo ""
  local partial_dir
  partial_dir="$(resolve_partial_dir)"
  if [[ -n "${partial_dir:-}" && -d "$partial_dir" ]]; then
    echo -e "${CYAN}当前 partial: $partial_dir${NC}"
    ls -lht "$partial_dir"/*.md 2>/dev/null || echo "  （空）"
  fi
  echo ""
  echo "AdaIN tensorboard 日志:"
  ls -lhtd "$SELF_DIR/logs"/* 2>/dev/null | head -8 || echo "暂无"
}

show_partial() {
  local target="${1:-}" partial_dir
  partial_dir="$(resolve_partial_dir)"
  if [[ -z "${partial_dir:-}" || ! -d "$partial_dir" ]]; then
    echo -e "${YELLOW}未找到 partial 目录${NC}"
    return 1
  fi

  echo -e "${CYAN}=== partial 目录: $partial_dir ===${NC}"
  echo ""

  if [[ -n "$target" ]]; then
    local f="$partial_dir/${target}.md"
    if [[ ! -f "$f" ]]; then
      echo -e "${RED}未找到: $f${NC}"
      return 1
    fi
    cat "$f"
    return 0
  fi

  local found=0
  for ds in "${DATASET_ORDER[@]}"; do
    local f="$partial_dir/${ds}.md"
    if [[ -f "$f" ]]; then
      found=1
      echo -e "${GREEN}--- $ds ---${NC}"
      grep -E '^\| '"$ds"' ' "$f" 2>/dev/null || true
      grep -E '^\| Accuracy \|' "$f" 2>/dev/null | head -1 || true
      echo ""
    else
      echo -e "${YELLOW}--- $ds --- 未完成${NC}"
      echo ""
    fi
  done

  if [[ $found -eq 0 ]]; then
    echo "尚无 partial 结果文件"
    ls -la "$partial_dir" 2>/dev/null || true
  fi
}

merge_partials() {
  echo -e "${GREEN}重新合并 partial → $RESULT_FILE${NC}"
  "$SELF_DIR/run_st_saclf_train_bat.sh" merge
}

show_result() {
  if [[ -f "$RESULT_FILE" ]]; then
    echo -e "${CYAN}=== $RESULT_FILE ===${NC}"
    echo ""
    cat "$RESULT_FILE"
  else
    echo -e "${YELLOW}未找到: $RESULT_FILE${NC}"
    echo "训练中可通过 partial 查看已完成数据集；全部完成后自动合并。"
    show_partial
  fi
}

case "${1:-help}" in
  start)   start_training   ;;
  stop)    stop_training    ;;
  restart) restart_training ;;
  status)  check_status     ;;
  tail)    tail_log         ;;
  logs)    list_logs        ;;
  partial) show_partial "${2:-}" ;;
  merge)   merge_partials   ;;
  result)  show_result      ;;
  help|-h|--help) show_help ;;
  *)       show_help; exit 1 ;;
esac

exit 0
