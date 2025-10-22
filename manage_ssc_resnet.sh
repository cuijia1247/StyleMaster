#!/bin/bash

# SSC ResNet 训练进程管理脚本
# Author: cuijia1247
# Date: 2025-10-22

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PID_FILE="ssc_resnet_train.pid"
LOG_DIR="log"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 显示使用帮助
show_help() {
    echo "SSC ResNet 训练进程管理工具"
    echo ""
    echo "用法: $0 {start|stop|restart|status|tail|logs}"
    echo ""
    echo "命令说明:"
    echo "  start   - 启动训练进程"
    echo "  stop    - 停止训练进程"
    echo "  restart - 重启训练进程"
    echo "  status  - 查看训练进程状态"
    echo "  tail    - 实时查看最新日志"
    echo "  logs    - 列出所有日志文件"
    echo ""
}

# 启动训练
start_training() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            echo -e "${YELLOW}训练进程已在运行 (PID: $PID)${NC}"
            return 1
        else
            rm -f "$PID_FILE"
        fi
    fi
    
    echo -e "${GREEN}启动ResNet训练进程...${NC}"
    ./run_ssc_resnet_background.sh
}

# 停止训练
stop_training() {
    if [ ! -f "$PID_FILE" ]; then
        echo -e "${YELLOW}未找到PID文件，训练进程可能未运行${NC}"
        return 1
    fi
    
    PID=$(cat "$PID_FILE")
    if ! ps -p "$PID" > /dev/null 2>&1; then
        echo -e "${YELLOW}训练进程未运行 (PID: $PID)${NC}"
        rm -f "$PID_FILE"
        return 1
    fi
    
    echo -e "${GREEN}停止训练进程 (PID: $PID)...${NC}"
    kill "$PID"
    
    # 等待进程结束
    for i in {1..10}; do
        if ! ps -p "$PID" > /dev/null 2>&1; then
            echo -e "${GREEN}训练进程已停止${NC}"
            rm -f "$PID_FILE"
            return 0
        fi
        sleep 1
    done
    
    # 如果进程仍在运行，强制结束
    if ps -p "$PID" > /dev/null 2>&1; then
        echo -e "${RED}进程未响应，强制结束...${NC}"
        kill -9 "$PID"
        rm -f "$PID_FILE"
    fi
}

# 重启训练
restart_training() {
    echo -e "${GREEN}重启训练进程...${NC}"
    stop_training
    sleep 2
    start_training
}

# 查看状态
check_status() {
    if [ ! -f "$PID_FILE" ]; then
        echo -e "${RED}训练进程未运行${NC}"
        return 1
    fi
    
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo -e "${GREEN}训练进程正在运行${NC}"
        echo "PID: $PID"
        echo ""
        echo "进程信息:"
        ps -fp "$PID"
        echo ""
        echo "GPU使用情况:"
        nvidia-smi 2>/dev/null || echo "nvidia-smi 不可用"
        echo ""
        
        # 显示最新日志文件
        LATEST_LOG=$(ls -t $LOG_DIR/ssc_resnet_*.log 2>/dev/null | head -1)
        if [ -n "$LATEST_LOG" ]; then
            echo "最新日志: $LATEST_LOG"
            echo "日志文件大小: $(du -h "$LATEST_LOG" | cut -f1)"
        fi
    else
        echo -e "${RED}训练进程未运行 (PID文件存在但进程不存在)${NC}"
        echo "清理PID文件..."
        rm -f "$PID_FILE"
        return 1
    fi
}

# 实时查看日志
tail_log() {
    LATEST_LOG=$(ls -t $LOG_DIR/ssc_resnet_*.log 2>/dev/null | head -1)
    if [ -z "$LATEST_LOG" ]; then
        echo -e "${RED}未找到日志文件${NC}"
        return 1
    fi
    
    echo -e "${GREEN}实时查看日志: $LATEST_LOG${NC}"
    echo "按 Ctrl+C 退出"
    echo ""
    tail -f "$LATEST_LOG"
}

# 列出所有日志
list_logs() {
    echo -e "${GREEN}所有ResNet训练日志:${NC}"
    echo ""
    if [ -d "$LOG_DIR" ]; then
        ls -lht $LOG_DIR/ssc_resnet_*.log 2>/dev/null | awk '{print $9, "(" $5 ")", $6, $7, $8}' || echo "未找到日志文件"
    else
        echo "日志目录不存在"
    fi
}

# 主程序
case "$1" in
    start)
        start_training
        ;;
    stop)
        stop_training
        ;;
    restart)
        restart_training
        ;;
    status)
        check_status
        ;;
    tail)
        tail_log
        ;;
    logs)
        list_logs
        ;;
    *)
        show_help
        exit 1
        ;;
esac

exit 0

