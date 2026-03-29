#!/bin/bash

# Transformer 超参数网格搜索进程管理脚本
# Author: cuijia1247
# Date: 2026-03-29
# 用法：从项目根目录执行 ./remote_sh/manage_transformer_param_optim.sh {start|stop|status|tail|tail-current|result|logs}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

PID_FILE="transformer_param_optim.pid"
RESULT_DIR="param_optim/transformer"
LOG_DIR="$RESULT_DIR/logs"
RESULT_CSV="$RESULT_DIR/grid_search_results.csv"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

show_help() {
    echo "Transformer 超参数网格搜索进程管理工具"
    echo ""
    echo "用法: $0 {start|stop|status|tail|tail-current|result|logs}"
    echo ""
    echo "命令说明:"
    echo "  start        - 启动参数搜索进程"
    echo "  stop         - 停止参数搜索进程"
    echo "  status       - 查看进程状态与搜索进度"
    echo "  tail         - 实时查看整体进度日志（nohup 输出）"
    echo "  tail-current - 实时查看当前正在训练组合的详细日志"
    echo "  result       - 查看已完成的汇总结果 CSV"
    echo "  logs         - 列出所有已生成的单组日志文件"
    echo ""
}

start_optim() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            echo -e "${YELLOW}参数搜索进程已在运行 (PID: $PID)${NC}"
            return 1
        else
            rm -f "$PID_FILE"
        fi
    fi
    echo -e "${GREEN}启动参数搜索进程...${NC}"
    ./remote_sh/run_transformer_param_optim.sh
}

stop_optim() {
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

    echo -e "${GREEN}停止参数搜索进程 (PID: $PID)...${NC}"
    kill "$PID"
    for i in {1..10}; do
        if ! ps -p "$PID" > /dev/null 2>&1; then
            echo -e "${GREEN}进程已停止${NC}"
            rm -f "$PID_FILE"
            return 0
        fi
        sleep 1
    done

    if ps -p "$PID" > /dev/null 2>&1; then
        echo -e "${RED}进程未响应，强制结束...${NC}"
        kill -9 "$PID"
        rm -f "$PID_FILE"
    fi
}

check_status() {
    echo -e "${CYAN}========== 进程状态 ==========${NC}"
    if [ ! -f "$PID_FILE" ]; then
        echo -e "${RED}参数搜索进程未运行${NC}"
    else
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            echo -e "${GREEN}参数搜索进程正在运行 (PID: $PID)${NC}"
            echo ""
            ps -fp "$PID"
        else
            echo -e "${RED}进程已结束（PID 文件残留，自动清理）${NC}"
            rm -f "$PID_FILE"
        fi
    fi

    echo ""
    echo -e "${CYAN}========== 搜索进度 ==========${NC}"
    if [ -f "$RESULT_CSV" ]; then
        TOTAL_LINES=$(wc -l < "$RESULT_CSV")
        DONE=$((TOTAL_LINES - 1))  # 减去 header 行
        echo "已完成组合数: $DONE"
        echo ""
        echo "最近 5 条结果:"
        tail -5 "$RESULT_CSV"
    else
        echo "尚无结果（CSV 文件未生成）"
    fi

    echo ""
    echo -e "${CYAN}========== GPU 状态 ==========${NC}"
    nvidia-smi 2>/dev/null || echo "nvidia-smi 不可用"
}

tail_nohup() {
    LATEST=$(ls -t "$RESULT_DIR"/nohup_*.log 2>/dev/null | head -1)
    if [ -z "$LATEST" ]; then
        echo -e "${RED}未找到 nohup 日志文件${NC}"
        return 1
    fi
    echo -e "${GREEN}实时查看整体进度: $LATEST${NC}"
    echo "按 Ctrl+C 退出"
    echo ""
    tail -f "$LATEST"
}

tail_current() {
    LATEST=$(ls -t "$LOG_DIR"/*.log 2>/dev/null | head -1)
    if [ -z "$LATEST" ]; then
        echo -e "${RED}未找到单组训练日志${NC}"
        return 1
    fi
    echo -e "${GREEN}实时查看当前训练日志: $LATEST${NC}"
    echo "按 Ctrl+C 退出"
    echo ""
    tail -f "$LATEST"
}

show_result() {
    if [ ! -f "$RESULT_CSV" ]; then
        echo -e "${RED}结果文件不存在: $RESULT_CSV${NC}"
        return 1
    fi
    echo -e "${GREEN}汇总结果 ($RESULT_CSV):${NC}"
    echo ""
    column -t -s ',' "$RESULT_CSV"
}

list_logs() {
    echo -e "${GREEN}单组训练日志列表 ($LOG_DIR):${NC}"
    echo ""
    ls -lht "$LOG_DIR"/*.log 2>/dev/null | awk '{print $9, "("$5")", $6, $7, $8}' || echo "暂无日志文件"
}

case "$1" in
    start)        start_optim   ;;
    stop)         stop_optim    ;;
    status)       check_status  ;;
    tail)         tail_nohup    ;;
    tail-current) tail_current  ;;
    result)       show_result   ;;
    logs)         list_logs     ;;
    *)            show_help; exit 1 ;;
esac

exit 0
