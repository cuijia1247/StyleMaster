#!/usr/bin/env bash
#
# SimCLR 批量训练进程 / 日志管理
# 用法: ./selfsupervised/manage_simclr_train_bat.sh {start|stop|restart|status|tail|logs|result}

set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SELF_DIR/.." && pwd)"
cd "$ROOT"

PID_FILE="$SELF_DIR/simclr_bat.pid"
LOG_DIR="$SELF_DIR/logs"
RESULT_FILE="$SELF_DIR/simclr_result.md"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

show_help() {
  echo "SimCLR 批量训练（六数据集 × 每库 3 次 run，mean±std 见 simclr_result.md）"
  echo ""
  echo "用法: $0 {start|stop|restart|status|tail|logs|result|help}"
  echo ""
  echo "  start   - 启动 run_simclr_train_bat.sh（nohup 后台）"
  echo "  stop    - 结束批量任务（读 PID 文件）"
  echo "  restart - stop 后 start"
  echo "  status  - 进程与 GPU、最新日志摘要"
  echo "  tail    - 实时查看最新 batch 日志（同 tail -f）"
  echo "  logs    - 列出 selfsupervised/logs 下 simclr_bat 日志"
  echo "  result  - 打印 simclr_result.md"
  echo ""
}

start_training() {
  if [[ -f "$PID_FILE" ]]; then
    PID="$(cat "$PID_FILE")"
    if ps -p "$PID" > /dev/null 2>&1; then
      echo -e "${YELLOW}SimCLR 批量任务已在运行 (PID: $PID)${NC}"
      return 1
    fi
    rm -f "$PID_FILE"
  fi
  echo -e "${GREEN}启动 SimCLR 批量训练...${NC}"
  "$SELF_DIR/run_simclr_train_bat.sh"
}

stop_training() {
  if [[ ! -f "$PID_FILE" ]]; then
    echo -e "${YELLOW}未找到 PID 文件，可能未在运行${NC}"
    return 1
  fi
  PID="$(cat "$PID_FILE")"
  if ! ps -p "$PID" > /dev/null 2>&1; then
    echo -e "${YELLOW}进程未运行 (PID: $PID)${NC}"
    rm -f "$PID_FILE"
    return 1
  fi
  echo -e "${GREEN}停止 SimCLR 批量任务 (PID: $PID)...${NC}"
  kill "$PID" || true
  for _ in {1..20}; do
    if ! ps -p "$PID" > /dev/null 2>&1; then
      echo -e "${GREEN}已停止${NC}"
      rm -f "$PID_FILE"
      return 0
    fi
    sleep 1
  done
  echo -e "${RED}强制结束...${NC}"
  kill -9 "$PID" 2>/dev/null || true
  rm -f "$PID_FILE"
}

restart_training() {
  if [[ -f "$PID_FILE" ]]; then
    stop_training || true
    sleep 2
  fi
  start_training
}

check_status() {
  if [[ ! -f "$PID_FILE" ]]; then
    echo -e "${RED}SimCLR 批量任务未运行（无 PID 文件）${NC}"
    return 1
  fi
  PID="$(cat "$PID_FILE")"
  if ps -p "$PID" > /dev/null 2>&1; then
    echo -e "${GREEN}SimCLR 批量任务运行中${NC}"
    echo "PID: $PID"
    ps -fp "$PID" 2>/dev/null || ps -p "$PID"
    echo ""
    nvidia-smi 2>/dev/null || echo "nvidia-smi 不可用"
    echo ""
    LATEST="$(ls -t "$LOG_DIR"/simclr_bat_*.log 2>/dev/null | head -1 || true)"
    if [[ -n "${LATEST:-}" ]]; then
      echo -e "${CYAN}最新日志: $LATEST ($(du -h "$LATEST" | cut -f1))${NC}"
      echo "最后 8 行:"
      tail -8 "$LATEST"
    fi
  else
    echo -e "${RED}PID 文件存在但进程已退出${NC}"
    rm -f "$PID_FILE"
    return 1
  fi
}

tail_log() {
  LATEST="$(ls -t "$LOG_DIR"/simclr_bat_*.log 2>/dev/null | head -1 || true)"
  if [[ -z "${LATEST:-}" ]]; then
    echo -e "${RED}未找到 $LOG_DIR/simclr_bat_*.log${NC}"
    return 1
  fi
  echo -e "${GREEN}tail -f $LATEST${NC} (Ctrl+C 退出)"
  tail -f "$LATEST"
}

list_logs() {
  echo -e "${GREEN}SimCLR 批量日志 ($LOG_DIR):${NC}"
  echo ""
  if [[ -d "$LOG_DIR" ]]; then
    ls -lht "$LOG_DIR"/simclr_bat_*.log 2>/dev/null || echo "暂无日志"
  else
    echo "目录不存在"
  fi
  echo ""
  echo "项目内 simclr 训练日志（Python 写入 log/）:"
  ls -lht "$ROOT/log"/simclr-resnet50-*.log 2>/dev/null | head -15 || echo "暂无"
}

show_result() {
  if [[ -f "$RESULT_FILE" ]]; then
    echo -e "${CYAN}=== $RESULT_FILE ===${NC}"
    echo ""
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
