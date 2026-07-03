#!/usr/bin/env bash
#
# OpenCLIP Community Variants 批量评测进程 / 日志管理
# 用法: ./CLIP-based/manage_openclip_train_bat.sh {start|stop|restart|status|tail|logs|result|help}

set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SELF_DIR/.." && pwd)"
cd "$ROOT"

PID_FILE="$SELF_DIR/openclip_bat.pid"
LASTLOG_FILE="$SELF_DIR/openclip_bat.lastlog"
LOG_DIR="$SELF_DIR/logs"
RESULT_FILE="$ROOT/ieee_access_paperdata/clip-based_multiple.md"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

show_help() {
  echo "OpenCLIP Community Variants 批量评测（ViT-H-14 冻结 CLIP）"
  echo "模式: zero-shot + linear_probe，五数据集 × runs=3，四项指标"
  echo "结果: ieee_access_paperdata/clip-based_multiple.md"
  echo ""
  echo "用法: $0 {start|stop|restart|status|tail|logs|result|help}"
  echo ""
  echo "  start   - 启动 run_openclip_train_bat.sh（nohup 后台）"
  echo "  stop    - 结束批量 shell 及训练子进程"
  echo "  restart - stop 后 start"
  echo "  status  - 进程、GPU、最新日志摘要"
  echo "  tail    - 实时查看最新 batch 日志"
  echo "  logs    - 列出 CLIP-based/logs 下 openclip_bat 日志"
  echo "  result  - 打印 ieee_access_paperdata/clip-based_multiple.md"
  echo ""
  echo "数据集: Painting91, Pandora, ArtBench, FashionStyle14, Arch"
}

start_training() {
  if [[ -f "$PID_FILE" ]]; then
    PID="$(cat "$PID_FILE")"
    if ps -p "$PID" > /dev/null 2>&1; then
      echo -e "${YELLOW}OpenCLIP 批量任务已在运行 (PID: $PID)${NC}"
      return 1
    fi
    rm -f "$PID_FILE"
  fi
  echo -e "${GREEN}启动 OpenCLIP 批量评测...${NC}"
  "$SELF_DIR/run_openclip_train_bat.sh"
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

  if pgrep -f "openclip_community_variants_train\.py" >/dev/null 2>&1; then
    echo -e "${GREEN}停止 openclip_community_variants_train.py 子进程...${NC}"
    pkill -f "openclip_community_variants_train\.py" 2>/dev/null || true
    sleep 2
    pkill -9 -f "openclip_community_variants_train\.py" 2>/dev/null || true
    stopped=1
  fi

  if [[ $stopped -eq 0 && ! -f "$PID_FILE" ]]; then
    echo -e "${YELLOW}未找到运行中的 OpenCLIP 批量任务${NC}"
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
      echo -e "${GREEN}OpenCLIP 批量 shell 运行中${NC}"
      echo "PID: $PID"
      ps -fp "$PID" 2>/dev/null || ps -p "$PID"
      echo ""
    else
      echo -e "${YELLOW}PID 文件存在但 shell 已退出 (PID: $PID)${NC}"
      rm -f "$PID_FILE"
    fi
  fi

  if pgrep -af "openclip_community_variants_train\.py" >/dev/null 2>&1; then
    running=1
    echo -e "${GREEN}openclip_community_variants_train.py 进程:${NC}"
    pgrep -af "openclip_community_variants_train\.py" || true
    echo ""
  fi

  if [[ $running -eq 0 ]]; then
    echo -e "${RED}OpenCLIP 批量任务未运行${NC}"
    return 1
  fi

  nvidia-smi 2>/dev/null || echo "nvidia-smi 不可用"
  echo ""

  LATEST=""
  if [[ -f "$LASTLOG_FILE" ]]; then
    LATEST="$(cat "$LASTLOG_FILE" 2>/dev/null || true)"
  fi
  if [[ -z "${LATEST:-}" || ! -f "$LATEST" ]]; then
    LATEST="$(ls -t "$LOG_DIR"/openclip_bat_*.log 2>/dev/null | head -1 || true)"
  fi
  if [[ -n "${LATEST:-}" ]]; then
    echo -e "${CYAN}最新日志: $LATEST ($(du -h "$LATEST" | cut -f1))${NC}"
    tail -12 "$LATEST"
  fi
}

tail_log() {
  LATEST=""
  if [[ -f "$LASTLOG_FILE" ]]; then
    LATEST="$(cat "$LASTLOG_FILE" 2>/dev/null || true)"
  fi
  if [[ -z "${LATEST:-}" || ! -f "$LATEST" ]]; then
    LATEST="$(ls -t "$LOG_DIR"/openclip_bat_*.log 2>/dev/null | head -1 || true)"
  fi
  if [[ -z "${LATEST:-}" ]]; then
    echo -e "${RED}未找到 $LOG_DIR/openclip_bat_*.log${NC}"
    return 1
  fi
  echo -e "${GREEN}tail -f $LATEST${NC} (Ctrl+C 退出)"
  tail -f "$LATEST"
}

list_logs() {
  echo -e "${GREEN}OpenCLIP 批量日志 ($LOG_DIR):${NC}"
  ls -lht "$LOG_DIR"/openclip_bat_*.log 2>/dev/null || echo "暂无"
  echo ""
  echo "部分结果目录:"
  ls -lhtd "$LOG_DIR"/openclip_partials_* 2>/dev/null | head -5 || echo "暂无"
  echo ""
  echo "训练日志 (log/oc-*.log):"
  ls -lht "$ROOT/log"/oc-*.log 2>/dev/null | head -10 || echo "暂无"
}

show_result() {
  if [[ -f "$RESULT_FILE" ]]; then
    echo -e "${CYAN}=== $RESULT_FILE ===${NC}"
    cat "$RESULT_FILE"
  else
    echo -e "${YELLOW}未找到: $RESULT_FILE${NC}"
  fi
}

case "${1:-help}" in
  start)   start_training   ;;
  stop)    stop_training    ;;
  restart) restart_training ;;
  status)  check_status     ;;
  tail)    tail_log         ;;
  logs)    list_logs        ;;
  result)  show_result      ;;
  help|-h|--help) show_help ;;
  *)       show_help; exit 1 ;;
esac

exit 0
