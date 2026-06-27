#!/usr/bin/env bash
#
# VICReg 后台训练管理：查看进度 / 停止 / 日志 / 结果
# 用法: ./selfsupervised/manage_vicreg_train.sh {status|progress|tail|logs|stop|start|restart|result|help}
#
# start  需通过 run_vicreg_train.sh 传入训练参数，例如:
#   ./selfsupervised/manage_vicreg_train.sh start -- --data_root /mnt/codes/data/style/Pandora --num_classes 12 --epochs 200 --runs 3 --merge_result

set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SELF_DIR/.." && pwd)"
cd "$ROOT"

PID_FILE="$SELF_DIR/vicreg_train.pid"
LASTLOG_FILE="$SELF_DIR/vicreg_train.lastlog"
LOG_DIR="$SELF_DIR/logs"
RESULT_FILE="$ROOT/ieee_access_paperdata/vicreg_multiple.md"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

_latest_log() {
  if [[ -f "$LASTLOG_FILE" ]]; then
    local p
    p="$(cat "$LASTLOG_FILE")"
    if [[ -f "$p" ]]; then
      echo "$p"
      return
    fi
  fi
  ls -t "$LOG_DIR"/vicreg_*.log 2>/dev/null | head -1 || true
}

show_help() {
  echo "VICReg 后台训练管理（vicreg_train.py）"
  echo ""
  echo "用法: $0 {status|progress|tail|logs|stop|start|restart|result|help}"
  echo ""
  echo "  status   - 进程、GPU、最新日志摘要"
  echo "  progress - 当前 epoch / 分类轮次 / 各 run 最佳指标"
  echo "  tail     - tail -f 最新后台日志"
  echo "  logs     - 列出 selfsupervised/logs 下 vicreg 日志"
  echo "  stop     - 停止后台任务"
  echo "  start    - 调用 run_vicreg_train.sh（参数放在 -- 之后）"
  echo "  restart  - stop 后 start"
  echo "  result   - 打印 ieee_access_paperdata/vicreg_multiple.md"
  echo ""
  echo "示例:"
  echo "  $0 start -- --data_root /mnt/codes/data/style/Pandora --num_classes 12 --epochs 200 --runs 3 --merge_result"
  echo ""
}

stop_training() {
  if [[ ! -f "$PID_FILE" ]]; then
    echo -e "${YELLOW}未找到 PID 文件 ($PID_FILE)，可能未在运行${NC}"
    return 1
  fi
  PID="$(cat "$PID_FILE")"
  if ! ps -p "$PID" > /dev/null 2>&1; then
    echo -e "${YELLOW}进程未运行 (PID: $PID)${NC}"
    rm -f "$PID_FILE"
    return 1
  fi
  echo -e "${GREEN}停止 VICReg 任务 (PID: $PID)...${NC}"
  kill "$PID" 2>/dev/null || true
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

start_training() {
  shift || true
  if [[ "${1:-}" == "--" ]]; then
    shift
  fi
  if [[ -f "$PID_FILE" ]]; then
    PID="$(cat "$PID_FILE")"
    if ps -p "$PID" > /dev/null 2>&1; then
      echo -e "${YELLOW}VICReg 任务已在运行 (PID: $PID)${NC}"
      return 1
    fi
    rm -f "$PID_FILE"
  fi
  echo -e "${GREEN}启动 VICReg 后台训练...${NC}"
  "$SELF_DIR/run_vicreg_train.sh" "$@"
}

restart_training() {
  stop_training || true
  sleep 2
  start_training "$@"
}

check_status() {
  LATEST="$(_latest_log)"
  if [[ ! -f "$PID_FILE" ]]; then
    echo -e "${RED}VICReg 后台任务未运行（无 PID 文件）${NC}"
    [[ -n "${LATEST:-}" ]] && echo -e "${CYAN}最近日志: $LATEST${NC}"
    return 1
  fi
  PID="$(cat "$PID_FILE")"
  if ps -p "$PID" > /dev/null 2>&1; then
    echo -e "${GREEN}VICReg 后台任务运行中${NC}"
    echo "PID: $PID"
    ps -fp "$PID" 2>/dev/null || ps -p "$PID"
    echo ""
    nvidia-smi 2>/dev/null || echo "nvidia-smi 不可用"
    echo ""
    if [[ -n "${LATEST:-}" ]]; then
      echo -e "${CYAN}日志: $LATEST ($(du -h "$LATEST" | cut -f1))${NC}"
      echo "最后 8 行:"
      tail -8 "$LATEST"
    fi
  else
    echo -e "${RED}PID 文件存在但进程已退出 (PID: $PID)${NC}"
    rm -f "$PID_FILE"
    return 1
  fi
}

show_progress() {
  LATEST="$(_latest_log)"
  if [[ -z "${LATEST:-}" ]]; then
    echo -e "${RED}未找到日志 ($LOG_DIR/vicreg_*.log)${NC}"
    return 1
  fi
  echo -e "${CYAN}=== 进度: $LATEST ===${NC}"
  echo ""

  if [[ -f "$PID_FILE" ]] && ps -p "$(cat "$PID_FILE")" > /dev/null 2>&1; then
    echo -e "${GREEN}状态: 运行中 (PID $(cat "$PID_FILE"))${NC}"
  else
    echo -e "${YELLOW}状态: 未运行或已结束${NC}"
  fi
  echo ""

  echo "--- 当前 run ---"
  rg "model name is" "$LATEST" 2>/dev/null | tail -1 || true
  rg "dataset = " "$LATEST" 2>/dev/null | tail -1 || true
  rg "The epoch is " "$LATEST" 2>/dev/null | tail -3 || true
  rg "classifer-train round is" "$LATEST" 2>/dev/null | tail -2 || true
  rg "Test result:" "$LATEST" 2>/dev/null | tail -2 || true
  echo ""

  echo "--- 各 run 最佳指标 (Best metrics) ---"
  rg "Best metrics:" "$LATEST" 2>/dev/null || echo "(尚无)"
  echo ""

  echo "--- 最近 vicreg 训练 loss ---"
  rg "vicreg train loss" "$LATEST" 2>/dev/null | tail -3 || true
}

tail_log() {
  LATEST="$(_latest_log)"
  if [[ -z "${LATEST:-}" ]]; then
    echo -e "${RED}未找到日志${NC}"
    return 1
  fi
  echo -e "${GREEN}tail -f $LATEST${NC} (Ctrl+C 退出)"
  tail -f "$LATEST"
}

list_logs() {
  echo -e "${GREEN}后台日志 ($LOG_DIR):${NC}"
  ls -lht "$LOG_DIR"/vicreg_*.log 2>/dev/null || echo "暂无"
  echo ""
  echo -e "${GREEN}Python 详细日志 ($ROOT/log/vicreg_*.log):${NC}"
  ls -lht "$ROOT/log"/vicreg_*.log 2>/dev/null | head -15 || echo "暂无"
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

CMD="${1:-help}"
shift || true

case "$CMD" in
  status)   check_status ;;
  progress) show_progress ;;
  tail)     tail_log ;;
  logs)     list_logs ;;
  stop)     stop_training ;;
  start)    start_training "$@" ;;
  restart)  restart_training "$@" ;;
  result)   show_result ;;
  help|-h|--help) show_help ;;
  *)        show_help; exit 1 ;;
esac

exit 0
