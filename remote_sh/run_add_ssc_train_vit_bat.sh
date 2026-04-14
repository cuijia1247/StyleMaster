#!/bin/bash

# add 版 SSC 批量多数据集训练脚本
# 策略: BarlowTwins + Cross-View SupCon 对齐损失 + StyleFusion 三路分类头
# 参数: epochs=40, base_lr=0.001(余弦退火), offset_bs=512,
#        cls_iteration=100, cls_lr=3e-5, cls_train_gap=2, iterations=1, runs=5（每数据集独立重复5次）
# 数据集: Painting91, Pandora, AVAstyle, Arch, FashionStyle14, WebStyle
# 结果记录: remote_sh/add_batch_result.md
# 用法: 从项目根目录执行 ./remote_sh/run_add_ssc_train_vit_bat.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

PID_FILE="add_bat_train.pid"
LOG_DIR="log"
RESULT_FILE="remote_sh/add_batch_result.md"
RUNNER_SCRIPT="remote_sh/_add_resnet50_bat_runner.py"

# 检查是否已有进程在运行
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "add 批量训练进程已在运行 (PID: $OLD_PID)"
        echo "如需重新启动，请先执行: ./remote_sh/manage_add_ssc_train_vit_bat.sh stop"
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

# 检查 GPU
python -c "import torch; print(f'CUDA 可用: {torch.cuda.is_available()}, 设备数: {torch.cuda.device_count()}')" 2>/dev/null || echo "警告: 无法检查 GPU"

# 生成 Python runner 脚本
cat > "$RUNNER_SCRIPT" << 'PYEOF'
# add 版 SSC 批量训练 runner（由 run_add_ssc_train_vit_bat.sh 生成）
# 策略：每数据集独立重复 RUNS 次（每次 iterations=1），关闭早停，记录每次 best_acc，
#       最终输出每数据集的 mean±std。
import sys
import os
import logging
import time
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import ssc_train_transformer_add

# ── 超参数 ─────────────────────────────────────────────────────────────────────
EPOCHS        = 40          # 每次跑 40 个 epoch
BASE_LR       = 0.001
OFFSET_BS     = 512
CLS_ITERATION = 100
CLASSIFIER_LR = 0.00003
CLS_TRAIN_GAP = 2
CLS_TEST_GAP  = 2
ITERATIONS    = 1           # 每次 SSCtrain 只跑 1 个 iteration
RUNS          = 5           # 每个数据集独立重复训练次数

# ── 数据集配置（名称 → 类别数）────────────────────────────────────────────────
DATASETS = [
    # ('Painting91',         13),
    # ('Pandora',            12),
    # ('AVAstyle',           14),
    # ('Arch',               25),
    # ('FashionStyle14',     14),
    ('webstyle', 10),
]

DATA_ROOT        = '/mnt/codes/data/style/'
MODEL_PATH       = os.path.join(ROOT, 'model') + '/'
PRE_FEATURE_PATH = os.path.join(ROOT, 'pretrainFeatures')
TRAINING_MODE    = 'original'
BASE_MODEL_PATH  = '###'

LOG_DIR     = os.path.join(ROOT, 'log')
RESULT_FILE = os.path.join(ROOT, 'remote_sh', 'add_batch_result.md')

os.makedirs(LOG_DIR, exist_ok=True)


def make_patched_parameter_load():
    """覆盖 parameter_load()，注入批量训练超参数"""
    def patched():
        return (
            EPOCHS,
            128,            # batch_size
            OFFSET_BS,
            BASE_LR,
            224,            # image_size
            CLS_ITERATION,
            CLASSIFIER_LR,
            '',             # model_name（由外部注入）
            CLS_TRAIN_GAP,
            'swin_base_patch4_window7_224',
            1024,           # ssc_input
            1024,           # ssc_output
            CLS_TEST_GAP,
        )
    return patched


def patch_no_early_stop_real():
    """禁用早停：将 SSCtrain 字节码中 es_patience=20 替换为 999999。
    
    es_patience 是 SSCtrain 内的局部变量（硬编码为 20），无法从外部直接覆盖，
    通过替换函数 __code__.co_consts 中的常量实现无侵入式禁用。
    """
    import ssc_train_transformer_add as _mod
    import types

    orig_fn = _mod.SSCtrain
    code    = orig_fn.__code__

    # 将字节码常量中的 es_patience 值 20 替换为 999999
    # co_consts 是 tuple，找到值为 20 的常量替换之
    new_consts = tuple(
        999999 if c == 20 else c
        for c in code.co_consts
    )
    new_code = code.replace(co_consts=new_consts)
    _mod.SSCtrain = types.FunctionType(
        new_code, orig_fn.__globals__, orig_fn.__name__,
        orig_fn.__defaults__, orig_fn.__closure__
    )
    return orig_fn


def restore_SSCtrain(orig_fn):
    import ssc_train_transformer_add as _mod
    _mod.SSCtrain = orig_fn


def patch_skip_last_save():
    """跳过末尾模型（-last.pth），只保留最佳模型（-best.pth）"""
    orig_save = torch.save

    def _save_best_only(obj, path, *args, **kwargs):
        if isinstance(path, str) and '-last.pth' in path:
            return
        orig_save(obj, path, *args, **kwargs)

    ssc_train_transformer_add.torch.save = _save_best_only
    return orig_save


def restore_save(orig_save):
    ssc_train_transformer_add.torch.save = orig_save


def setup_logger(log_file):
    logger = logging.getLogger(f'add_bat_{time.time()}')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger, fh, sh


def init_result_file():
    """初始化或追加结果 Markdown 文件"""
    header = (
        f"参数: epochs={EPOCHS}, base_lr={BASE_LR}(余弦退火), offset_bs={OFFSET_BS}, "
        f"cls_iteration={CLS_ITERATION}, cls_lr={CLASSIFIER_LR}, "
        f"cls_train_gap={CLS_TRAIN_GAP}, cls_test_gap={CLS_TEST_GAP}, "
        f"iterations={ITERATIONS}, runs={RUNS}(每数据集重复次数), 早停=禁用, "
        f"loss=BarlowTwins+CrossViewSupCon(lam=0.05,T=1.0), "
        f"classifier=StyleFusion(256+512+256→1024)"
    )
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    with open(RESULT_FILE, 'a', encoding='utf-8') as f:
        f.write(f"\n---\n## 批次开始: {ts}\n\n_{header}_\n\n")
        # 明细表：每次 run 单独一行
        f.write("### 各次明细\n\n")
        f.write("| 数据集 | Run | 最佳准确率 | 训练时长(min) | 状态 |\n")
        f.write("|--------|-----|-----------|--------------|------|\n")


def append_run_result(dataset_name, run_no, best_acc, elapsed_min, status):
    """写入单次 run 结果"""
    label = dataset_name.replace('webstyle/subImages', 'WebStyle')
    with open(RESULT_FILE, 'a', encoding='utf-8') as f:
        f.write(f"| {label} | {run_no} | {best_acc:.4f} "
                f"| {elapsed_min:.1f} | {status} |\n")


def append_summary(dataset_name, accs, total_min):
    """写入单数据集的均值±标准差汇总行"""
    label    = dataset_name.replace('webstyle/subImages', 'WebStyle')
    arr      = np.array([a for a in accs if a >= 0])
    mean_acc = arr.mean() if len(arr) > 0 else 0.0
    std_acc  = arr.std()  if len(arr) > 0 else 0.0
    with open(RESULT_FILE, 'a', encoding='utf-8') as f:
        f.write(f"\n**{label}** 汇总: "
                f"`{mean_acc:.4f} ± {std_acc:.4f}` "
                f"(runs={len(arr)}, total={total_min:.1f}min, "
                f"acc_list={[f'{a:.4f}' for a in accs]})\n\n")


def append_final_summary(all_results):
    """所有数据集完成后写入汇总对比表"""
    with open(RESULT_FILE, 'a', encoding='utf-8') as f:
        f.write("\n### 全数据集汇总\n\n")
        f.write("| 数据集 | Mean Acc | Std | 各次准确率 |\n")
        f.write("|--------|---------|-----|----------|\n")
        for ds_name, accs in all_results:
            label = ds_name.replace('webstyle/subImages', 'WebStyle')
            arr   = np.array([a for a in accs if a >= 0])
            mean_ = arr.mean() if len(arr) > 0 else 0.0
            std_  = arr.std()  if len(arr) > 0 else 0.0
            acc_s = ' / '.join(f'{a:.4f}' for a in accs)
            f.write(f"| {label} | {mean_:.4f} | {std_:.4f} | {acc_s} |\n")


def parse_best_accuracy(log_file):
    """从日志中解析 best_accuracy（'The best accuracy is X' 行）"""
    best_acc = 0.0
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            if 'best accuracy is' in line and 'last accuracy is' in line:
                try:
                    seg      = line.split('best accuracy is')[1]
                    best_acc = float(seg.split(',')[0].strip())
                except (IndexError, ValueError):
                    pass
    return best_acc


def run_once(dataset_name, class_num, run_no):
    """对单个数据集执行一次完整训练（iterations=1, epochs=EPOCHS, 无早停）"""
    safe_name    = dataset_name.replace('/', '_')
    current_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    model_name   = f'ssc-add-{safe_name}-ep{EPOCHS}-run{run_no}'
    feature_name = f'{safe_name}_vit'
    log_file     = os.path.join(LOG_DIR, f'ssc-add-{safe_name}-{current_time}.log')

    logger, fh, sh = setup_logger(log_file)
    logger.info('=' * 80)
    logger.info('[%s] Run %d/%d 开始  class_num=%d', dataset_name, run_no, RUNS, class_num)
    logger.info('epochs=%d, iterations=%d(无早停), base_lr=%s, cls_lr=%s',
                EPOCHS, ITERATIONS, BASE_LR, CLASSIFIER_LR)
    logger.info('=' * 80)

    ssc_train_transformer_add.parameter_load = make_patched_parameter_load()
    orig_fn   = patch_no_early_stop_real()
    orig_save = patch_skip_last_save()

    data_source = os.path.join(DATA_ROOT, dataset_name) + '/'
    t_start     = time.time()
    best_acc    = 0.0
    status      = 'OK'

    try:
        ssc_train_transformer_add.SSCtrain(
            logger, MODEL_PATH, current_time, model_name,
            data_source, class_num, ITERATIONS,
            TRAINING_MODE, BASE_MODEL_PATH,
            PRE_FEATURE_PATH, feature_name
        )
        best_acc = parse_best_accuracy(log_file)
    except Exception as e:
        logger.error('训练异常: %s', str(e), exc_info=True)
        status   = f'ERR:{type(e).__name__}'
        best_acc = -1.0
    finally:
        restore_SSCtrain(orig_fn)
        restore_save(orig_save)
        logger.removeHandler(fh)
        logger.removeHandler(sh)
        fh.close()

    elapsed_min = (time.time() - t_start) / 60.0
    return best_acc, elapsed_min, status


def main():
    init_result_file()
    all_results = []   # [(ds_name, [acc1, acc2, ...])]

    total_ds = len(DATASETS)
    for ds_idx, (ds_name, cls_num) in enumerate(DATASETS, start=1):
        print(f"\n{'='*60}")
        print(f"[{ds_idx}/{total_ds}] 数据集: {ds_name}  (共 {RUNS} 次运行，早停已禁用)")
        print(f"{'='*60}")

        accs      = []
        total_min = 0.0

        for run_no in range(1, RUNS + 1):
            print(f"  >> Run {run_no}/{RUNS} 开始 ...")
            best_acc, elapsed_min, status = run_once(ds_name, cls_num, run_no)
            accs.append(best_acc)
            total_min += elapsed_min
            append_run_result(ds_name, run_no, best_acc, elapsed_min, status)
            print(f"  << Run {run_no}/{RUNS} 完成  best_acc={best_acc:.4f}  "
                  f"{elapsed_min:.1f}min  [{status}]")

        # 单数据集汇总
        arr  = np.array([a for a in accs if a >= 0])
        mean_ = arr.mean() if len(arr) > 0 else 0.0
        std_  = arr.std()  if len(arr) > 0 else 0.0
        append_summary(ds_name, accs, total_min)
        all_results.append((ds_name, accs))
        print(f"\n  [{ds_name}] {RUNS} 次结果: {[f'{a:.4f}' for a in accs]}")
        print(f"  均值 ± 标准差: {mean_:.4f} ± {std_:.4f}")

    # 全数据集汇总表
    append_final_summary(all_results)
    print(f"\n全部完成，结果已写入: {RESULT_FILE}")


if __name__ == '__main__':
    main()
PYEOF

# 启动后台训练
TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S")
LOG_FILE="$LOG_DIR/add_bat_main_${TIMESTAMP}.log"

echo "=========================================="
echo "启动 ADD 版 SSC 批量训练 (6 个数据集)"
echo "主日志: $LOG_FILE"
echo "结果文件: $RESULT_FILE"
echo "开始时间: $(date)"
echo "=========================================="

nohup python "$RUNNER_SCRIPT" > "$LOG_FILE" 2>&1 &
TRAIN_PID=$!
echo "$TRAIN_PID" > "$PID_FILE"

echo "训练已启动，PID: $TRAIN_PID"
echo ""
echo "管理命令:"
echo "  状态/日志/停止: ./remote_sh/manage_add_ssc_train_vit_bat.sh {status|tail|stop}"
echo ""

sleep 3
if ps -p "$TRAIN_PID" > /dev/null 2>&1; then
    echo "√ 训练进程启动成功，可安全断开 SSH"
else
    echo "✗ 进程启动失败，请查看日志: $LOG_FILE"
    rm -f "$PID_FILE"
    exit 1
fi
