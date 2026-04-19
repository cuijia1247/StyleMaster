#!/bin/bash

# ssc_train_densenet169.py 批量：DenseNet169-3ch + ssc.utils + classifier_enhance.EfficientClassifier
# 六数据集各 RUNS=5 次，记录每次 best acc 与 mean±std → remote_sh/ssc_densenet169_batch_result.md
# 用法: 项目根目录 ./remote_sh/run_ssc_train_densenet_bat.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

PID_FILE="ssc_d169_bat_train.pid"
LOG_DIR="log"
RESULT_FILE="remote_sh/ssc_densenet169_batch_result.md"
RUNNER_SCRIPT="remote_sh/_ssc_train_densenet169_bat_runner.py"

if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "ssc_train_densenet169 批量已在运行 (PID: $OLD_PID)"
        echo "停止: ./remote_sh/manage_ssc_train_densenet_bat.sh stop"
        exit 1
    else
        rm -f "$PID_FILE"
    fi
fi

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

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}')" 2>/dev/null || echo "警告: 无法检查 GPU"

cat > "$RUNNER_SCRIPT" << 'PYEOF'
# ssc_train_densenet169 批量 runner（由 run_ssc_train_densenet_bat.sh 生成）
import sys
import os
import logging
import time
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import ssc_train_densenet169

# 训练/分类器超参：ssc_train_densenet169.parameter_load()；外层迭代：DEFAULT_SSC_OUTER_ITERATIONS
RUNS = 5

DATASETS = [
    ('Painting91',         13),
    ('Pandora',            12),
    ('AVAstyle',           14),
    ('Arch',               25),
    ('FashionStyle14',     14),
    ('webstyle/subImages', 10),
]

DATA_ROOT        = '/mnt/codes/data/style/'
MODEL_PATH       = os.path.join(ROOT, 'model') + '/'
TRAINING_MODE    = 'original'
BASE_MODEL_PATH  = os.path.join(ROOT, 'model', 'base-best.pth')

LOG_DIR     = os.path.join(ROOT, 'log')
RESULT_FILE = os.path.join(ROOT, 'remote_sh', 'ssc_densenet169_batch_result.md')

os.makedirs(LOG_DIR, exist_ok=True)


def _unpack_params():
    """与 ssc_train_densenet169.parameter_load() 返回值顺序一致。"""
    (
        epochs,
        batch_size_,
        offset_bs_,
        base_lr_,
        image_size_,
        classifier_iteration_,
        classifier_lr_,
        model_name_,
        classifier_training_gap_,
        ssc_input_,
        ssc_output_,
        classifier_test_gap_,
        backbone_cache_workers_,
        dataloader_num_workers_,
        classifier_cache_k_,
    ) = ssc_train_densenet169.parameter_load()
    return {
        'epochs': epochs,
        'batch_size': batch_size_,
        'offset_bs': offset_bs_,
        'base_lr': base_lr_,
        'image_size': image_size_,
        'classifier_iteration': classifier_iteration_,
        'classifier_lr': classifier_lr_,
        'classifier_training_gap': classifier_training_gap_,
        'ssc_input': ssc_input_,
        'ssc_output': ssc_output_,
        'classifier_test_gap': classifier_test_gap_,
        'backbone_cache_workers': backbone_cache_workers_,
        'dataloader_num_workers': dataloader_num_workers_,
        'classifier_cache_k': classifier_cache_k_,
    }


def patch_no_early_stop_real():
    import ssc_train_densenet169 as _mod
    import types
    orig_fn = _mod.SSCtrain
    code    = orig_fn.__code__
    new_consts = tuple(999999 if c == 20 else c for c in code.co_consts)
    new_code = code.replace(co_consts=new_consts)
    _mod.SSCtrain = types.FunctionType(
        new_code, orig_fn.__globals__, orig_fn.__name__,
        orig_fn.__defaults__, orig_fn.__closure__
    )
    return orig_fn


def restore_SSCtrain(orig_fn):
    import ssc_train_densenet169 as _mod
    _mod.SSCtrain = orig_fn


def patch_skip_last_save():
    orig_save = torch.save

    def _save_best_only(obj, path, *args, **kwargs):
        if isinstance(path, str) and '-last.pth' in path:
            return
        orig_save(obj, path, *args, **kwargs)

    ssc_train_densenet169.torch.save = _save_best_only
    return orig_save


def restore_save(orig_save):
    ssc_train_densenet169.torch.save = orig_save


def setup_logger(log_file):
    logger = logging.getLogger(f'ssc_d169_bat_{time.time()}')
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
    p = _unpack_params()
    header = (
        f"ssc_train_densenet169.parameter_load(): "
        f"epochs={p['epochs']}, batch={p['batch_size']}, offset_bs={p['offset_bs']}, "
        f"base_lr={p['base_lr']}, image_size={p['image_size']}, "
        f"classifier_iteration={p['classifier_iteration']}, classifier_lr={p['classifier_lr']}, "
        f"classifier_train_gap={p['classifier_training_gap']}, classifier_test_gap={p['classifier_test_gap']}, "
        f"ssc_io={p['ssc_input']}/{p['ssc_output']}, "
        f"backbone_cache_workers={p['backbone_cache_workers']}, "
        f"dataloader_num_workers={p['dataloader_num_workers']}, classifier_cache_k={p['classifier_cache_k']}; "
        f"SSCtrain outer_iterations={ssc_train_densenet169.DEFAULT_SSC_OUTER_ITERATIONS}, "
        f"batch_runs={RUNS}, 早停=禁用"
    )
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    with open(RESULT_FILE, 'a', encoding='utf-8') as f:
        f.write(f"\n---\n## 批次: {ts}\n\n_{header}_\n\n")
        f.write("### 各次明细\n\n")
        f.write("| 数据集 | Run | 最佳准确率 | 训练时长(min) | 状态 |\n")
        f.write("|--------|-----|-----------|--------------|------|\n")


def append_run_result(dataset_name, run_no, best_acc, elapsed_min, status):
    label = dataset_name.replace('webstyle/subImages', 'WebStyle')
    with open(RESULT_FILE, 'a', encoding='utf-8') as f:
        f.write(f"| {label} | {run_no} | {best_acc:.4f} "
                f"| {elapsed_min:.1f} | {status} |\n")


def append_summary(dataset_name, accs, total_min):
    label    = dataset_name.replace('webstyle/subImages', 'WebStyle')
    arr      = np.array([a for a in accs if a >= 0])
    mean_acc = arr.mean() if len(arr) > 0 else 0.0
    std_acc  = arr.std(ddof=1) if len(arr) > 1 else 0.0
    with open(RESULT_FILE, 'a', encoding='utf-8') as f:
        f.write(f"\n**{label}** 汇总: `{mean_acc:.4f} ± {std_acc:.4f}` "
                f"(runs={len(arr)}, total={total_min:.1f}min, "
                f"acc_list={[f'{a:.4f}' for a in accs]})\n\n")


def append_final_summary(all_results):
    with open(RESULT_FILE, 'a', encoding='utf-8') as f:
        f.write("\n### 全数据集汇总\n\n")
        f.write("| 数据集 | Mean Acc | Std | 各次准确率 |\n")
        f.write("|--------|---------|-----|----------|\n")
        for ds_name, accs in all_results:
            label = ds_name.replace('webstyle/subImages', 'WebStyle')
            arr   = np.array([a for a in accs if a >= 0])
            mean_ = arr.mean() if len(arr) > 0 else 0.0
            std_  = arr.std(ddof=1) if len(arr) > 1 else 0.0
            acc_s = ' / '.join(f'{a:.4f}' for a in accs)
            f.write(f"| {label} | {mean_:.4f} | {std_:.4f} | {acc_s} |\n")


def parse_best_accuracy(log_file):
    best_acc = 0.0
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            if 'best accuracy is' in line.lower() and 'last accuracy is' in line.lower():
                try:
                    seg = line.lower().split('best accuracy is')[1]
                    best_acc = float(seg.split(',')[0].strip())
                except (IndexError, ValueError):
                    pass
    return best_acc


def run_once(dataset_name, class_num, run_no):
    p = _unpack_params()
    safe_name    = dataset_name.replace('/', '_')
    current_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    model_name   = f'ssc-d169-{safe_name}-ep{p["epochs"]}-run{run_no}'
    log_file     = os.path.join(LOG_DIR, f'ssc-d169-{safe_name}-{current_time}.log')

    logger, fh, sh = setup_logger(log_file)
    logger.info('=' * 80)
    logger.info('[%s] Run %d/%d  class_num=%d', dataset_name, run_no, RUNS, class_num)
    logger.info(
        'parameter_load: epochs=%d, cls_iter=%d, cls_lr=%g, cls_cache_k=%d | '
        'SSCtrain iterations=%d',
        p['epochs'], p['classifier_iteration'], p['classifier_lr'],
        p['classifier_cache_k'], ssc_train_densenet169.DEFAULT_SSC_OUTER_ITERATIONS,
    )
    logger.info('=' * 80)

    orig_fn   = patch_no_early_stop_real()
    orig_save = patch_skip_last_save()

    data_source = os.path.join(DATA_ROOT, dataset_name) + '/'
    t_start     = time.time()
    best_acc    = 0.0
    status      = 'OK'

    try:
        ssc_train_densenet169.SSCtrain(
            logger, MODEL_PATH, current_time, model_name,
            data_source, class_num, ssc_train_densenet169.DEFAULT_SSC_OUTER_ITERATIONS,
            TRAINING_MODE, BASE_MODEL_PATH,
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
    all_results = []

    total_ds = len(DATASETS)
    for ds_idx, (ds_name, cls_num) in enumerate(DATASETS, start=1):
        print(f"\n{'='*60}")
        print(f"[{ds_idx}/{total_ds}] {ds_name}  ({RUNS} 次)")
        print(f"{'='*60}")

        accs      = []
        total_min = 0.0

        for run_no in range(1, RUNS + 1):
            print(f"  >> Run {run_no}/{RUNS} ...")
            best_acc, elapsed_min, status = run_once(ds_name, cls_num, run_no)
            accs.append(best_acc)
            total_min += elapsed_min
            append_run_result(ds_name, run_no, best_acc, elapsed_min, status)
            print(f"  << Run {run_no}/{RUNS}  best_acc={best_acc:.4f}  "
                  f"{elapsed_min:.1f}min  [{status}]")

        arr   = np.array([a for a in accs if a >= 0])
        mean_ = arr.mean() if len(arr) > 0 else 0.0
        std_  = arr.std(ddof=1) if len(arr) > 1 else 0.0
        append_summary(ds_name, accs, total_min)
        all_results.append((ds_name, accs))
        print(f"\n  [{ds_name}] {mean_:.4f} ± {std_:.4f}")

    append_final_summary(all_results)
    print(f"\n结果: {RESULT_FILE}")


if __name__ == '__main__':
    main()
PYEOF

TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S")
LOG_FILE="$LOG_DIR/ssc_d169_bat_main_${TIMESTAMP}.log"

echo "=========================================="
echo "ssc_train_densenet169 批量 (6 数据集 × 5 次)"
echo "主日志: $LOG_FILE"
echo "结果: $RESULT_FILE"
echo "开始: $(date)"
echo "=========================================="

nohup python "$RUNNER_SCRIPT" > "$LOG_FILE" 2>&1 &
TRAIN_PID=$!
echo "$TRAIN_PID" > "$PID_FILE"

echo "PID: $TRAIN_PID"
echo "管理: ./remote_sh/manage_ssc_train_densenet_bat.sh {status|tail|stop|result}"
echo ""

sleep 3
if ps -p "$TRAIN_PID" > /dev/null 2>&1; then
    echo "√ 已启动"
else
    echo "✗ 失败，见: $LOG_FILE"
    rm -f "$PID_FILE"
    exit 1
fi
