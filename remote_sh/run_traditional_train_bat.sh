#!/bin/bash

# Traditional backbone 批量训练脚本
# 策略: 冻结预训练 backbone 提取特征，训练轻量 Classifier
# Backbone: vgg16, vgg19, resnet50, resnet101, inception_v3, vit_b_16, vit_l_16
# 数据集: Painting91 (13 类), runs=5, cls_iteration=100
# 结果记录: remote_sh/traditional_train_bat_result.md
# 最佳模型: remote_sh/traditional_train_best/
# 用法: 从项目根目录执行 ./remote_sh/run_traditional_train_bat.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

PID_FILE="traditional_bat_train.pid"
LOG_DIR="log"
RESULT_FILE="remote_sh/traditional_train_bat_result.md"
RUNNER_SCRIPT="remote_sh/_traditional_bat_runner.py"

# 检查是否已有进程在运行
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "traditional 批量训练进程已在运行 (PID: $OLD_PID)"
        echo "如需重新启动，请先执行: ./remote_sh/manage_traditional_train_bat.sh stop"
        exit 1
    else
        rm -f "$PID_FILE"
    fi
fi

# 检查 conda 环境
if ! command -v conda &> /dev/null; then
    echo "错误: conda 未安装或未在 PATH 中"
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ssc

if [ "$CONDA_DEFAULT_ENV" != "ssc" ]; then
    echo "错误: 无法激活 conda 环境 'ssc'"
    exit 1
fi

mkdir -p "$LOG_DIR"
mkdir -p "remote_sh/traditional_train_best"

# 检查 GPU
python -c "import torch; print(f'CUDA 可用: {torch.cuda.is_available()}, 设备数: {torch.cuda.device_count()}')" 2>/dev/null || echo "警告: 无法检查 GPU"

# 生成 Python runner 脚本
cat > "$RUNNER_SCRIPT" << 'PYEOF'
# Traditional backbone 批量训练 runner（由 run_traditional_train_bat.sh 生成）
# 对 7 个 backbone × 6 个数据集逐一训练（runs=5），
# 记录每个 backbone 的 mean±std，并将最佳 classifier 模型保存至 traditional_train_best/。
import sys
import os
import glob
import time
import shutil
import subprocess
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ── 配置 ──────────────────────────────────────────────────────────────────────
BACKBONES = ['vgg16', 'vgg19', 'resnet50', 'resnet101',
             'inception_v3', 'vit_b_16', 'vit_l_16']

# (数据集文件夹名, 类别数)
DATASETS = [
    ('Painting91',    13),
    ('Pandora',       12),
    ('AVAstyle',      14),
    ('Arch',          25),
    ('FashionStyle14', 14),
    ('webstyle',      10),
]

DATA_BASE       = '/mnt/codes/data/style'
RUNS            = 5
CLASSIFIER_LR   = 3e-4
CLASSIFIER_ITER = 100
BATCH_SIZE      = 64
NUM_WORKERS     = 4

LOG_DIR     = os.path.join(ROOT, 'log')
MODEL_DIR   = os.path.join(ROOT, 'model')
BEST_DIR    = os.path.join(ROOT, 'remote_sh', 'traditional_train_best')
RESULT_FILE = os.path.join(ROOT, 'remote_sh', 'traditional_train_bat_result.md')

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(BEST_DIR, exist_ok=True)


def init_result_file():
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    header = (
        f"runs={RUNS}, cls_iteration={CLASSIFIER_ITER}, "
        f"cls_lr={CLASSIFIER_LR}, batch_size={BATCH_SIZE}"
    )
    ds_str = ', '.join(f'{n}({c}类)' for n, c in DATASETS)
    with open(RESULT_FILE, 'a', encoding='utf-8') as f:
        f.write(f"\n---\n## 批次开始: {ts}\n\n_{header}_\n\n")
        f.write(f"_数据集: {ds_str}_\n\n")


def begin_dataset_section(dataset_name):
    with open(RESULT_FILE, 'a', encoding='utf-8') as f:
        f.write(f"\n### {dataset_name}\n\n")
        f.write("| Backbone | Mean Acc | Std | 各次准确率 | 训练时长(min) | 状态 |\n")
        f.write("|----------|---------|-----|-----------|--------------|------|\n")


def append_backbone_result(backbone, mean_acc, std_acc, acc_list, elapsed_min, status):
    acc_str = ' / '.join(f'{a:.4f}' for a in acc_list) if acc_list else '-'
    with open(RESULT_FILE, 'a', encoding='utf-8') as f:
        f.write(f"| {backbone} | {mean_acc:.4f} | {std_acc:.4f} "
                f"| {acc_str} | {elapsed_min:.1f} | {status} |\n")


def append_final_summary(all_results):
    """写入全数据集 × 全 backbone 汇总矩阵（mean acc）"""
    header_cols = ' | '.join(BACKBONES)
    sep_cols    = ' | '.join(['-------'] * len(BACKBONES))
    with open(RESULT_FILE, 'a', encoding='utf-8') as f:
        f.write("\n---\n### 全数据集 × Backbone 汇总 (Mean Acc)\n\n")
        f.write(f"| 数据集 | {header_cols} |\n")
        f.write(f"|--------|{sep_cols}|\n")
        for ds_name, bb_results in all_results:
            # bb_results: {backbone: mean_acc}
            vals = ' | '.join(
                f'{bb_results.get(bb, 0.0):.4f}' for bb in BACKBONES
            )
            f.write(f"| {ds_name} | {vals} |\n")
        ts = time.strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"\n_批次结束: {ts}_\n")


def copy_best_model(backbone, dataset_name, start_ts):
    """从 model/ 中找到本次运行新生成的 .pth，按 best_acc 选最优，复制至 BEST_DIR。"""
    import torch
    pattern    = os.path.join(MODEL_DIR, f'traditional-{backbone}-{dataset_name}-*.pth')
    candidates = [f for f in glob.glob(pattern) if os.path.getmtime(f) >= start_ts]
    if not candidates:
        return None

    best_pth, best_val = None, -1.0
    for f in candidates:
        try:
            ckpt = torch.load(f, map_location='cpu', weights_only=True)
            acc  = float(ckpt.get('best_acc', 0))
            if acc > best_val:
                best_val, best_pth = acc, f
        except Exception:
            pass

    if best_pth:
        acc_int = int(round(best_val * 10000))
        dest = os.path.join(BEST_DIR, f'{dataset_name}-{backbone}-best-acc{acc_int}.pth')
        shutil.copy2(best_pth, dest)
        return dest
    return None


def parse_results_from_log(log_file):
    """从日志文件中解析 mean_acc, std_acc 和各次 best_acc 列表。"""
    mean_acc, std_acc, acc_list = 0.0, 0.0, []
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            if 'Final result (mean+-std):' in line:
                seg   = line.split('Final result (mean+-std):')[1].strip()
                parts = seg.split('+-')
                try:
                    mean_acc = float(parts[0].strip())
                    std_acc  = float(parts[1].strip())
                except (IndexError, ValueError):
                    pass
            if '5-run best acc list:' in line:
                seg = line.split('5-run best acc list:')[1].strip()
                try:
                    acc_list = [float(x) for x in seg.strip('[]').split(',')]
                except ValueError:
                    pass
    return mean_acc, std_acc, acc_list


def run_backbone(backbone, dataset_name, num_classes):
    """对单个 (backbone, dataset) 组合完整运行 traditional_train.py。"""
    start_ts = time.time()
    ts_str   = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(start_ts))
    log_file = os.path.join(LOG_DIR, f'traditional-{dataset_name}-{backbone}-{ts_str}.log')
    data_root = os.path.join(DATA_BASE, dataset_name)

    cmd = [
        sys.executable, os.path.join(ROOT, 'traditional_train.py'),
        '--backbone',             backbone,
        '--data_root',            data_root,
        '--num_classes',          str(num_classes),
        '--runs',                 str(RUNS),
        '--classifier_iteration', str(CLASSIFIER_ITER),
        '--classifier_lr',        str(CLASSIFIER_LR),
        '--batch_size',           str(BATCH_SIZE),
        '--num_workers',          str(NUM_WORKERS),
    ]

    status = 'OK'
    try:
        with open(log_file, 'w', encoding='utf-8') as lf:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=ROOT
            )
            # 逐行同时写入子日志和 stdout（bat_main 日志通过 nohup 捕获 stdout）
            for raw in proc.stdout:
                text = raw.decode('utf-8', errors='replace')
                lf.write(text)
                sys.stdout.write(text)
                sys.stdout.flush()
            proc.wait()
        if proc.returncode != 0:
            status = f'ERR:rc={proc.returncode}'
    except Exception as e:
        status = f'ERR:{type(e).__name__}'

    elapsed_min              = (time.time() - start_ts) / 60.0
    mean_acc, std_acc, acc_list = parse_results_from_log(log_file)
    best_dest                = copy_best_model(backbone, dataset_name, start_ts)

    print(f'  [{dataset_name}] backbone={backbone}  mean={mean_acc:.4f}  '
          f'std={std_acc:.4f}  {elapsed_min:.1f}min  [{status}]')
    if best_dest:
        print(f'  最佳模型已保存: {best_dest}')

    return mean_acc, std_acc, acc_list, elapsed_min, status


def main():
    init_result_file()
    # all_results: [(ds_name, {backbone: mean_acc})]
    all_results = []

    total_ds = len(DATASETS)
    for ds_idx, (ds_name, num_classes) in enumerate(DATASETS, start=1):
        print(f"\n{'='*70}")
        print(f"[数据集 {ds_idx}/{total_ds}] {ds_name}  ({num_classes} 类)")
        print(f"{'='*70}")

        begin_dataset_section(ds_name)
        bb_mean = {}

        for bb_idx, backbone in enumerate(BACKBONES, start=1):
            print(f"\n  [{bb_idx}/{len(BACKBONES)}] Backbone: {backbone}")
            mean_acc, std_acc, acc_list, elapsed_min, status = \
                run_backbone(backbone, ds_name, num_classes)
            append_backbone_result(backbone, mean_acc, std_acc,
                                   acc_list, elapsed_min, status)
            bb_mean[backbone] = mean_acc

        all_results.append((ds_name, bb_mean))

    append_final_summary(all_results)
    print(f"\n全部完成，结果已写入: {RESULT_FILE}")
    print(f"最佳模型目录: {BEST_DIR}")


if __name__ == '__main__':
    main()
PYEOF

# 启动后台训练
TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S")
LOG_FILE="$LOG_DIR/traditional_bat_main_${TIMESTAMP}.log"

echo "=========================================="
echo "启动 Traditional backbone 批量训练 (6 数据集 × 7 backbone)"
echo "数据集: Painting91 / Pandora / AVAstyle / Arch / FashionStyle14 / webstyle"
echo "主日志: $LOG_FILE"
echo "结果文件: $RESULT_FILE"
echo "最佳模型: remote_sh/traditional_train_best/"
echo "开始时间: $(date)"
echo "=========================================="

nohup python "$RUNNER_SCRIPT" > "$LOG_FILE" 2>&1 &
TRAIN_PID=$!
echo "$TRAIN_PID" > "$PID_FILE"

echo "训练已启动，PID: $TRAIN_PID"
echo ""
echo "管理命令:"
echo "  状态/日志/停止: ./remote_sh/manage_traditional_train_bat.sh {status|tail|stop}"
echo ""

sleep 3
if ps -p "$TRAIN_PID" > /dev/null 2>&1; then
    echo "√ 训练进程启动成功，可安全断开 SSH"
else
    echo "✗ 进程启动失败，请查看日志: $LOG_FILE"
    rm -f "$PID_FILE"
    exit 1
fi
