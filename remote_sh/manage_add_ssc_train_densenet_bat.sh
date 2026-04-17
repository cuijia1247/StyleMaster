#!/bin/bash

# DenseNet169 add 版 SSC 批量训练进程管理
# 用法: ./remote_sh/manage_add_ssc_train_densenet_bat.sh {start|stop|restart|status|tail|logs|result}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

PID_FILE="densenet_bat_train.pid"
LOG_DIR="log"
RESULT_FILE="remote_sh/densenet_batch_result.md"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

show_help() {
    echo "DenseNet169 add 版 SSC 批量训练管理"
    echo ""
    echo "用法: $0 {start|stop|restart|status|tail|logs|result}"
    echo ""
    echo "  start   - 启动（run_add_ssc_train_densenet_bat.sh）"
    echo "  stop    - 停止"
    echo "  restart - 重启"
    echo "  status  - 进程与 GPU"
    echo "  tail    - 主日志"
    echo "  logs    - 列出相关日志"
    echo "  result  - 查看 densenet_batch_result.md"
    echo ""
}

start_training() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            echo -e "${YELLOW}批量训练已在运行 (PID: $PID)${NC}"
            return 1
        else
            rm -f "$PID_FILE"
        fi
    fi
    echo -e "${GREEN}启动 DenseNet169 批量训练...${NC}"
    ./remote_sh/run_add_ssc_train_densenet_bat.sh
}

stop_training() {
    if [ ! -f "$PID_FILE" ]; then
        echo -e "${YELLOW}未找到 PID 文件${NC}"
        return 1
    fi
    PID=$(cat "$PID_FILE")
    if ! ps -p "$PID" > /dev/null 2>&1; then
        echo -e "${YELLOW}进程未运行 (PID: $PID)${NC}"
        rm -f "$PID_FILE"
        return 1
    fi
    echo -e "${GREEN}停止训练 (PID: $PID)...${NC}"
    kill "$PID"
    for i in {1..15}; do
        if ! ps -p "$PID" > /dev/null 2>&1; then
            echo -e "${GREEN}已停止${NC}"
            rm -f "$PID_FILE"
            return 0
        fi
        sleep 1
    done
    echo -e "${RED}强制结束...${NC}"
    kill -9 "$PID"
    rm -f "$PID_FILE"
}

restart_training() {
    echo -e "${GREEN}重启 DenseNet169 批量训练...${NC}"
    stop_training
    sleep 2
    start_training
}

check_status() {
    if [ ! -f "$PID_FILE" ]; then
        echo -e "${RED}批量训练未运行${NC}"
        return 1
    fi
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo -e "${GREEN}批量训练运行中${NC}"
        echo "PID: $PID"
        echo ""
        ps -fp "$PID"
        echo ""
        nvidia-smi 2>/dev/null || echo "nvidia-smi 不可用"
        echo ""
        LATEST=$(ls -t "$LOG_DIR"/densenet_bat_main_*.log \
                        "$LOG_DIR"/ssc-add-d169-*.log 2>/dev/null | head -1)
        if [ -n "$LATEST" ]; then
            echo -e "${CYAN}最新日志: $LATEST ($(du -h "$LATEST" | cut -f1))${NC}"
            tail -5 "$LATEST"
        fi
    else
        echo -e "${RED}PID 存在但进程未运行${NC}"
        rm -f "$PID_FILE"
        return 1
    fi
}

tail_log() {
    LATEST=$(ls -t "$LOG_DIR"/densenet_bat_main_*.log \
                    "$LOG_DIR"/ssc-add-d169-*.log 2>/dev/null | head -1)
    if [ -z "$LATEST" ]; then
        echo -e "${RED}未找到日志${NC}"
        return 1
    fi
    echo -e "${GREEN}tail -f $LATEST${NC} (Ctrl+C 退出)"
    tail -f "$LATEST"
}

list_logs() {
    echo -e "${GREEN}DenseNet 批量日志 (densenet_bat_main_* / ssc-add-d169-*):${NC}"
    echo ""
    if [ -d "$LOG_DIR" ]; then
        ls -lht "$LOG_DIR"/densenet_bat_main_*.log \
                "$LOG_DIR"/ssc-add-d169-*.log 2>/dev/null \
            | awk '{print $9, "("$5")", $6, $7, $8}' || echo "未找到日志"
    else
        echo "无 log 目录"
    fi
}

show_result() {
    if [ -f "$RESULT_FILE" ]; then
        echo -e "${CYAN}=== $RESULT_FILE ===${NC}"
        echo ""
        cat "$RESULT_FILE"
    else
        echo -e "${YELLOW}尚无结果文件: $RESULT_FILE${NC}"
    fi
}

case "$1" in
    start)   start_training   ;;
    stop)    stop_training    ;;
    restart) restart_training ;;
    status)  check_status     ;;
    tail)    tail_log         ;;
    logs)    list_logs        ;;
    result)  show_result      ;;
    *)       show_help; exit 1 ;;
esac

exit 0
