# DenseNet169 add 版 SSC 批量训练 runner（由 run_add_ssc_train_densenet_bat.sh 生成）
import sys
import os
import logging
import time
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import ssc_train_densnet169_add

# 与 ssc_train_densnet169_add.parameter_load() 默认一致，可按需改
EPOCHS                  = 35
BASE_LR                 = 0.001
OFFSET_BS               = 512
IMAGE_SIZE              = 224
CLS_ITERATION           = 100
CLASSIFIER_LR           = 0.00004
CLS_TRAIN_GAP           = 2
CLS_TEST_GAP            = 2
ITERATIONS              = 1
RUNS                    = 3
BACKBONE_CACHE_WORKERS  = 8
DATALOADER_NUM_WORKERS  = 8
CLASSIFIER_CACHE_K      = 12

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
RESULT_FILE = os.path.join(ROOT, 'remote_sh', 'densenet_batch_result.md')

os.makedirs(LOG_DIR, exist_ok=True)


def make_patched_parameter_load():
    def patched():
        return (
            EPOCHS,
            128,
            OFFSET_BS,
            BASE_LR,
            IMAGE_SIZE,
            CLS_ITERATION,
            CLASSIFIER_LR,
            '',
            CLS_TRAIN_GAP,
            1664,
            1664,
            CLS_TEST_GAP,
            BACKBONE_CACHE_WORKERS,
            DATALOADER_NUM_WORKERS,
            CLASSIFIER_CACHE_K,
        )
    return patched


def patch_no_early_stop_real():
    import ssc_train_densnet169_add as _mod
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
    import ssc_train_densnet169_add as _mod
    _mod.SSCtrain = orig_fn


def patch_skip_last_save():
    orig_save = torch.save

    def _save_best_only(obj, path, *args, **kwargs):
        if isinstance(path, str) and '-last.pth' in path:
            return
        orig_save(obj, path, *args, **kwargs)

    ssc_train_densnet169_add.torch.save = _save_best_only
    return orig_save


def restore_save(orig_save):
    ssc_train_densnet169_add.torch.save = orig_save


def setup_logger(log_file):
    logger = logging.getLogger(f'd169_bat_{time.time()}')
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
    header = (
        f"参数: DenseNet169-6ch add, epochs={EPOCHS}, base_lr={BASE_LR}(余弦), offset_bs={OFFSET_BS}, "
        f"cls_iteration={CLS_ITERATION}, cls_lr={CLASSIFIER_LR}, "
        f"cls_train_gap={CLS_TRAIN_GAP}, cls_test_gap={CLS_TEST_GAP}, "
        f"iterations={ITERATIONS}, runs={RUNS}, 早停=禁用, "
        f"ssc_io=1664, backbone_cache_workers={BACKBONE_CACHE_WORKERS}, "
        f"dataloader_num_workers={DATALOADER_NUM_WORKERS}, classifier_cache_k={CLASSIFIER_CACHE_K}, "
        f"classifier=EfficientRWPClassifier"
    )
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    with open(RESULT_FILE, 'a', encoding='utf-8') as f:
        f.write(f"\n---\n## 批次开始: {ts}\n\n_{header}_\n\n")
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
        f.write(f"\n**{label}** 汇总: "
                f"`{mean_acc:.4f} ± {std_acc:.4f}` "
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
    safe_name    = dataset_name.replace('/', '_')
    current_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    model_name   = f'ssc-add-d169-{safe_name}-ep{EPOCHS}-run{run_no}'
    log_file     = os.path.join(LOG_DIR, f'ssc-add-d169-{safe_name}-{current_time}.log')

    logger, fh, sh = setup_logger(log_file)
    logger.info('=' * 80)
    logger.info('[%s] Run %d/%d  class_num=%d', dataset_name, run_no, RUNS, class_num)
    logger.info('epochs=%d, iterations=%d, base_lr=%s, cls_lr=%s',
                EPOCHS, ITERATIONS, BASE_LR, CLASSIFIER_LR)
    logger.info('=' * 80)

    ssc_train_densnet169_add.parameter_load = make_patched_parameter_load()
    orig_fn   = patch_no_early_stop_real()
    orig_save = patch_skip_last_save()

    data_source = os.path.join(DATA_ROOT, dataset_name) + '/'
    t_start     = time.time()
    best_acc    = 0.0
    status      = 'OK'

    try:
        ssc_train_densnet169_add.SSCtrain(
            logger, MODEL_PATH, current_time, model_name,
            data_source, class_num, ITERATIONS,
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
        print(f"[{ds_idx}/{total_ds}] 数据集: {ds_name}  ({RUNS} 次)")
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
        print(f"\n  [{ds_name}] accs: {[f'{a:.4f}' for a in accs]}")
        print(f"  mean ± std: {mean_:.4f} ± {std_:.4f}")

    append_final_summary(all_results)
    print(f"\n全部完成，结果已写入: {RESULT_FILE}")


if __name__ == '__main__':
    main()
