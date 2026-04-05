#!/bin/bash

# ViT 批量训练进程管理脚本
# 用法: ./remote_sh/manage_ssc_train_vit_bat.sh {start|stop|restart|status|tail|logs|result}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

PID_FILE="vit_bat_train.pid"
LOG_DIR="log"
RESULT_FILE="remote_sh/vit_batch_result.md"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

show_help() {
    echo "ViT 批量训练进程管理工具"
    echo ""
    echo "用法: $0 {start|stop|restart|status|tail|logs|result}"
    echo ""
    echo "  start   - 启动批量训练"
    echo "  stop    - 停止批量训练"
    echo "  restart - 重启批量训练"
    echo "  status  - 查看进程状态与 GPU 占用"
    echo "  tail    - 实时查看主日志"
    echo "  logs    - 列出所有训练日志"
    echo "  result  - 查看 Markdown 结果汇总"
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
    echo -e "${GREEN}启动批量训练...${NC}"
    ./remote_sh/run_ssc_train_vit_bat.sh
}

stop_training() {
    if [ ! -f "$PID_FILE" ]; then
        echo -e "${YELLOW}未找到 PID 文件，进程可能未运行${NC}"
        return 1
    fi

    PID=$(cat "$PID_FILE")
    if ! ps -p "$PID" > /dev/null 2>&1; then
        echo -e "${YELLOW}进程未运行 (PID: $PID)${NC}"
        rm -f "$PID_FILE"
        return 1
    fi

    echo -e "${GREEN}停止训练进程 (PID: $PID)...${NC}"
    kill "$PID"

    for i in {1..15}; do
        if ! ps -p "$PID" > /dev/null 2>&1; then
            echo -e "${GREEN}进程已停止${NC}"
            rm -f "$PID_FILE"
            return 0
        fi
        sleep 1
    done

    echo -e "${RED}进程未响应，强制结束...${NC}"
    kill -9 "$PID"
    rm -f "$PID_FILE"
}

restart_training() {
    echo -e "${GREEN}重启批量训练...${NC}"
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
        echo -e "${GREEN}批量训练正在运行${NC}"
        echo "PID: $PID"
        echo ""
        echo "进程信息:"
        ps -fp "$PID"
        echo ""
        echo "GPU 使用情况:"
        nvidia-smi 2>/dev/null || echo "nvidia-smi 不可用"
        echo ""
        # 最新主日志
        LATEST=$(ls -t "$LOG_DIR"/vit_bat_main_*.log "$LOG_DIR"/vit-bat-*.log 2>/dev/null | head -1)
        if [ -n "$LATEST" ]; then
            echo -e "${CYAN}最新日志: $LATEST ($(du -h "$LATEST" | cut -f1))${NC}"
            echo "最后 5 行:"
            tail -5 "$LATEST"
        fi
    else
        echo -e "${RED}PID 文件存在但进程未运行${NC}"
        rm -f "$PID_FILE"
        return 1
    fi
}

tail_log() {
    # 优先找主进程日志（vit_bat_main_），其次找单数据集日志（vit-bat-）
    LATEST=$(ls -t "$LOG_DIR"/vit_bat_main_*.log "$LOG_DIR"/vit-bat-*.log 2>/dev/null | head -1)
    if [ -z "$LATEST" ]; then
        echo -e "${RED}未找到批量训练日志文件${NC}"
        return 1
    fi
    echo -e "${GREEN}实时查看: $LATEST${NC} (Ctrl+C 退出)"
    echo ""
    tail -f "$LATEST"
}

list_logs() {
    echo -e "${GREEN}批量训练日志列表 (vit_bat_main_* / vit-bat-*):${NC}"
    echo ""
    if [ -d "$LOG_DIR" ]; then
        ls -lht "$LOG_DIR"/vit_bat_main_*.log "$LOG_DIR"/vit-bat-*.log 2>/dev/null \
            | awk '{print $9, "("$5")", $6, $7, $8}' || echo "未找到日志"
    else
        echo "日志目录不存在"
    fi
}

show_result() {
    if [ -f "$RESULT_FILE" ]; then
        echo -e "${CYAN}=== 结果汇总: $RESULT_FILE ===${NC}"
        echo ""
        cat "$RESULT_FILE"
    else
        echo -e "${YELLOW}结果文件尚不存在: $RESULT_FILE${NC}"
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
