#!/usr/bin/env bash
#
# Barlow Twins 批量训练进程 / 日志管理（barlowtwins_train.py × 五数据集）
# 用法: ./selfsupervised/manage_barlowtwins_train_bat.sh {start|stop|restart|status|tail|logs|result|help}

set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SELF_DIR/.." && pwd)"
cd "$ROOT"

PID_FILE="$SELF_DIR/barlowtwins_bat.pid"
LASTLOG_FILE="$SELF_DIR/barlowtwins_bat.lastlog"
LOG_DIR="$SELF_DIR/logs"
RESULT_FILE="$ROOT/ieee_access_paperdata/BarlowTwins_multiple.md"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

show_help() {
  echo "Barlow Twins 批量训练（barlowtwins_train.py × 五数据集 × runs=3）"
  echo "结果: ieee_access_paperdata/BarlowTwins_multiple.md（四项指标，格式对齐 vgg16_multiple.md）"
  echo ""
  echo "用法: $0 {start|stop|restart|status|tail|logs|result|help}"
  echo ""
  echo "  start   - 启动 run_barlowtwins_train_bat.sh（nohup 后台）"
  echo "  stop    - 结束批量 shell 及 barlowtwins_train.py 子进程"
  echo "  restart - stop 后 start"
  echo "  status  - 进程、GPU、最新日志摘要"
  echo "  tail    - 实时查看最新 batch 日志"
  echo "  logs    - 列出 selfsupervised/logs 下 barlowtwins_bat 日志"
  echo "  result  - 打印 ieee_access_paperdata/BarlowTwins_multiple.md"
  echo ""
  echo "数据集: Painting91, Pandora, ArtBench, FashionStyle14, Arch"
  echo "训练脚本: $ROOT/selfsupervised/barlowtwins_train.py（默认超参，--runs 3）"
}

start_training() {
  if [[ -f "$PID_FILE" ]]; then
    PID="$(cat "$PID_FILE")"
    if ps -p "$PID" > /dev/null 2>&1; then
      echo -e "${YELLOW}Barlow Twins 批量任务已在运行 (PID: $PID)${NC}"
      return 1
    fi
    rm -f "$PID_FILE"
  fi
  echo -e "${GREEN}启动 Barlow Twins 批量训练 (barlowtwins_train.py)...${NC}"
  "$SELF_DIR/run_barlowtwins_train_bat.sh"
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

  if pgrep -f "barlowtwins_train\.py" >/dev/null 2>&1; then
    echo -e "${GREEN}停止 barlowtwins_train.py 子进程...${NC}"
    pkill -f "barlowtwins_train\.py" 2>/dev/null || true
    sleep 2
    pkill -9 -f "barlowtwins_train\.py" 2>/dev/null || true
    stopped=1
  fi

  if [[ $stopped -eq 0 && ! -f "$PID_FILE" ]]; then
    echo -e "${YELLOW}未找到运行中的 Barlow Twins 批量任务${NC}"
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
      echo -e "${GREEN}Barlow Twins 批量 shell 运行中${NC}"
      echo "PID: $PID"
      ps -fp "$PID" 2>/dev/null || ps -p "$PID"
      echo ""
    else
      echo -e "${YELLOW}PID 文件存在但 shell 已退出 (PID: $PID)${NC}"
      rm -f "$PID_FILE"
    fi
  fi

  if pgrep -af "barlowtwins_train\.py" >/dev/null 2>&1; then
    running=1
    echo -e "${GREEN}barlowtwins_train.py 进程:${NC}"
    pgrep -af "barlowtwins_train\.py" || true
    echo ""
  fi

  if [[ $running -eq 0 ]]; then
    echo -e "${RED}Barlow Twins 批量任务未运行${NC}"
    return 1
  fi

  nvidia-smi 2>/dev/null || echo "nvidia-smi 不可用"
  echo ""

  LATEST=""
  if [[ -f "$LASTLOG_FILE" ]]; then
    LATEST="$(cat "$LASTLOG_FILE" 2>/dev/null || true)"
  fi
  if [[ -z "${LATEST:-}" || ! -f "$LATEST" ]]; then
    LATEST="$(ls -t "$LOG_DIR"/barlowtwins_bat_*.log 2>/dev/null | head -1 || true)"
  fi
  if [[ -n "${LATEST:-}" ]]; then
    echo -e "${CYAN}最新日志: $LATEST ($(du -h "$LATEST" | cut -f1))${NC}"
    echo "最后 12 行:"
    tail -12 "$LATEST"
  fi
}

tail_log() {
  LATEST=""
  if [[ -f "$LASTLOG_FILE" ]]; then
    LATEST="$(cat "$LASTLOG_FILE" 2>/dev/null || true)"
  fi
  if [[ -z "${LATEST:-}" || ! -f "$LATEST" ]]; then
    LATEST="$(ls -t "$LOG_DIR"/barlowtwins_bat_*.log 2>/dev/null | head -1 || true)"
  fi
  if [[ -z "${LATEST:-}" ]]; then
    echo -e "${RED}未找到 $LOG_DIR/barlowtwins_bat_*.log${NC}"
    return 1
  fi
  echo -e "${GREEN}tail -f $LATEST${NC} (Ctrl+C 退出)"
  tail -f "$LATEST"
}

list_logs() {
  echo -e "${GREEN}Barlow Twins 批量日志 ($LOG_DIR):${NC}"
  echo ""
  if [[ -d "$LOG_DIR" ]]; then
    ls -lht "$LOG_DIR"/barlowtwins_bat_*.log 2>/dev/null || echo "暂无 barlowtwins_bat 日志"
    echo ""
    echo "部分结果目录:"
    ls -lhtd "$LOG_DIR"/barlowtwins_partials_* 2>/dev/null | head -5 || echo "暂无"
    echo ""
    echo "epoch 耗时记录:"
    ls -lht "$LOG_DIR"/barlowtwins_partials_*/epoch_times/*.txt 2>/dev/null | head -10 || echo "暂无"
  else
    echo "目录不存在"
  fi
  echo ""
  echo "barlowtwins_train 训练日志 (log/):"
  ls -lht "$ROOT/log"/barlowtwins-resnet50-*.log 2>/dev/null | head -15 || echo "暂无"
}

show_result() {
  if [[ -f "$RESULT_FILE" ]]; then
    echo -e "${CYAN}=== $RESULT_FILE ===${NC}"
    echo ""
    cat "$RESULT_FILE"
  else
    echo -e "${YELLOW}未找到: $RESULT_FILE${NC}"
    echo "（全部数据集跑完后由 run_barlowtwins_train_bat.sh 合并生成）"
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
