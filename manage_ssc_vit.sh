#!/bin/bash

# SSC Transformer 训练管理脚本
# Author: cuijia1247
# Date: 2025-01-27

# 设置脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PID_FILE="ssc_train.pid"
LOG_DIR="log"

# 显示帮助信息
show_help() {
    echo "SSC Transformer 训练管理脚本"
    echo ""
    echo "用法: $0 [命令]"
    echo ""
    echo "可用命令:"
    echo "  status     - 查看训练状态"
    echo "  start      - 启动训练 (等同于 ./run_ssc_background.sh)"
    echo "  stop       - 停止训练"
    echo "  restart    - 重启训练"
    echo "  logs       - 查看最新日志 (最后50行)"
    echo "  tail       - 实时查看日志"
    echo "  list       - 列出所有相关进程"
    echo "  clean      - 清理日志文件"
    echo "  help       - 显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 status    # 查看训练状态"
    echo "  $0 tail      # 实时查看日志"
    echo "  $0 stop      # 停止训练"
}

# 检查训练状态
check_status() {
    if [ ! -f "$PID_FILE" ]; then
        echo "状态: 未运行 (无PID文件)"
        return 1
    fi
    
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "状态: 正在运行"
        echo "进程ID: $PID"
        echo "启动时间: $(ps -o lstart= -p "$PID" 2>/dev/null || echo "未知")"
        
        # 显示GPU使用情况
        if command -v nvidia-smi &> /dev/null; then
            echo ""
            echo "GPU使用情况:"
            nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | grep -E "^1" || echo "GPU 1 信息获取失败"
        fi
        
        return 0
    else
        echo "状态: 已停止 (进程不存在)"
        rm -f "$PID_FILE"
        return 1
    fi
}

# 启动训练
start_training() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            echo "训练已在运行 (PID: $PID)"
            return 1
        fi
    fi
    
    echo "启动训练..."
    ./run_ssc_background.sh
}

# 停止训练
stop_training() {
    if [ ! -f "$PID_FILE" ]; then
        echo "没有找到PID文件，训练可能未运行"
        return 1
    fi
    
    PID=$(cat "$PID_FILE")
    if ! ps -p "$PID" > /dev/null 2>&1; then
        echo "进程不存在，清理PID文件"
        rm -f "$PID_FILE"
        return 1
    fi
    
    echo "停止训练进程 (PID: $PID)..."
    kill "$PID"
    
    # 等待进程结束
    for i in {1..10}; do
        if ! ps -p "$PID" > /dev/null 2>&1; then
            echo "训练已停止"
            rm -f "$PID_FILE"
            return 0
        fi
        sleep 1
    done
    
    # 强制终止
    echo "强制终止进程..."
    kill -9 "$PID" 2>/dev/null
    rm -f "$PID_FILE"
    echo "训练已强制停止"
}

# 重启训练
restart_training() {
    echo "重启训练..."
    stop_training
    sleep 2
    start_training
}

# 查看最新日志
show_logs() {
    if [ ! -d "$LOG_DIR" ]; then
        echo "日志目录不存在: $LOG_DIR"
        return 1
    fi
    
    LATEST_LOG=$(ls -t "$LOG_DIR"/ssc_background_*.log 2>/dev/null | head -1)
    if [ -z "$LATEST_LOG" ]; then
        echo "没有找到日志文件"
        return 1
    fi
    
    echo "最新日志文件: $LATEST_LOG"
    echo "最后50行内容:"
    echo "----------------------------------------"
    tail -50 "$LATEST_LOG"
}

# 实时查看日志
tail_logs() {
    if [ ! -d "$LOG_DIR" ]; then
        echo "日志目录不存在: $LOG_DIR"
        return 1
    fi
    
    LATEST_LOG=$(ls -t "$LOG_DIR"/ssc_background_*.log 2>/dev/null | head -1)
    if [ -z "$LATEST_LOG" ]; then
        echo "没有找到日志文件"
        return 1
    fi
    
    echo "实时查看日志: $LATEST_LOG"
    echo "按 Ctrl+C 退出"
    echo "----------------------------------------"
    tail -f "$LATEST_LOG"
}

# 列出相关进程
list_processes() {
    echo "SSC相关进程:"
    echo "----------------------------------------"
    
    # 查找Python训练进程
    ps aux | grep -E "(ssc_train_transformer\.py|python.*ssc_train_transformer)" | grep -v grep || echo "没有找到SSC训练进程"
    
    echo ""
    echo "GPU进程:"
    echo "----------------------------------------"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi pmon -c 1 2>/dev/null || echo "无法获取GPU进程信息"
    else
        echo "nvidia-smi 不可用"
    fi
}

# 清理日志文件
clean_logs() {
    if [ ! -d "$LOG_DIR" ]; then
        echo "日志目录不存在: $LOG_DIR"
        return 1
    fi
    
    LOG_COUNT=$(ls "$LOG_DIR"/ssc_background_*.log 2>/dev/null | wc -l)
    if [ "$LOG_COUNT" -eq 0 ]; then
        echo "没有日志文件需要清理"
        return 0
    fi
    
    echo "找到 $LOG_COUNT 个日志文件"
    echo "是否删除所有日志文件? (y/N)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        rm -f "$LOG_DIR"/ssc_background_*.log
        echo "日志文件已清理"
    else
        echo "取消清理"
    fi
}

# 主函数
main() {
    case "${1:-help}" in
        "status")
            check_status
            ;;
        "start")
            start_training
            ;;
        "stop")
            stop_training
            ;;
        "restart")
            restart_training
            ;;
        "logs")
            show_logs
            ;;
        "tail")
            tail_logs
            ;;
        "list")
            list_processes
            ;;
        "clean")
            clean_logs
            ;;
        "help"|*)
            show_help
            ;;
    esac
}

# 运行主函数
main "$@"
