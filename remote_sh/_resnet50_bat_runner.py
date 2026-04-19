# ResNet50 批量训练 runner（由 run_ssc_train_resnet_bat.sh 每次启动时生成；超参见 ssc_train_resnet_copy.parameter_load）
# 维护：修改批量逻辑请编辑本 shell 的 PYEOF，并与 remote_sh/_resnet50_bat_runner.py 保持一致后一并提交。
import sys
import os
import logging
import time
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import ssc_train_resnet_copy as ssc_train_resnet

# ── 固定配置 ─────────────────────────────────────────────────
# 训练超参一律来自 ssc_train_resnet_copy.parameter_load()，本文件不再覆盖。
# ITERATIONS：parameter_load 未定义；与 ssc_train_resnet_copy.parse_train_args --iterations 默认一致（1）。
# WebStyle：数据根为 DATA_ROOT/webstyle（与 traditional / MCCFNet 等一致），非 webstyle/subImages。
DATASETS = [
    ('Painting91',         13),
    ('Pandora',            12),
    ('AVAstyle',           14),
    ('Arch',               25),
    ('FashionStyle14',     14),
    ('webstyle',           10),
]

DATA_ROOT        = '/mnt/codes/data/style/'
MODEL_PATH       = os.path.join(ROOT, 'model') + '/'
PRE_FEATURE_PATH = os.path.join(ROOT, 'pretrainFeatures')
ITERATIONS       = 1
# 每数据集独立完整训练轮数；与 SSCtrain 的 dataset_repeat_runs 一致（每轮只计 best，汇总 mean±std）
DATASET_REPEAT_RUNS = 5
TRAINING_MODE    = 'original'
BASE_MODEL_PATH  = '###'

LOG_DIR     = os.path.join(ROOT, 'log')   # 与 ssc_train_resnet_copy __main__ 一致
RESULT_FILE = os.path.join(ROOT, 'remote_sh', 'resnet50_batch_result.md')

os.makedirs(LOG_DIR, exist_ok=True)


def _hparams_from_copy():
    """
    读取 parameter_load() 元组字段，与 SSCtrain 内解包顺序一致（仅用于日志/命名/结果表头）。
    """
    t = ssc_train_resnet.parameter_load()
    return {
        'epochs': t[0],
        'batch_size': t[1],
        'base_lr': t[3],
        'image_size': t[4],
        'cls_iter': t[5],
        'cls_lr': t[6],
        'cls_train_gap': t[9],
        'cls_test_gap': t[14],
    }


def patch_skip_last_save():
    """
    替换训练模块内的 torch.save，
    跳过文件名含 '-last.pth' 的末尾模型，只保留最佳模型（'-best.pth'）。
    返回原始 save 函数供 finally 恢复。
    """
    orig_save = torch.save

    def _save_best_only(obj, path, *args, **kwargs):
        if isinstance(path, str) and '-last.pth' in path:
            return
        orig_save(obj, path, *args, **kwargs)

    # 只替换训练模块内引用的 torch，不影响全局
    ssc_train_resnet.torch.save = _save_best_only
    return orig_save


def restore_save(orig_save):
    ssc_train_resnet.torch.save = orig_save


def setup_logger(log_file):
    """独立 logger，避免 handler 累积。"""
    logger = logging.getLogger(f"bat_{time.time()}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s - %(message)s")

    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger, fh, sh


def write_batch_result_header():
    """每次启动完整批量时覆盖写入表头（每库 DATASET_REPEAT_RUNS 轮 best + mean±std）。"""
    hp = _hparams_from_copy()
    with open(RESULT_FILE, 'w', encoding='utf-8') as f:
        f.write("# ResNet50 批量训练结果\n\n")
        f.write(
            "每数据集独立重复完整训练 **"
            + str(DATASET_REPEAT_RUNS)
            + "** 次；每轮仅记录该轮 **best** 测试准确率，**不**汇总 last。"
            " 末列 **mean±std** 为该库各轮 best 的均值 ± 样本标准差。\n\n"
        )
        f.write("超参来源: `ssc_train_resnet_copy.parameter_load()`\n\n")
        f.write(
            f"- epochs={hp['epochs']}, base_lr={hp['base_lr']}, image_size={hp['image_size']}\n"
            f"- cls_iteration={hp['cls_iter']}, cls_lr={hp['cls_lr']}, "
            f"cls_train_gap={hp['cls_train_gap']}, cls_test_gap={hp['cls_test_gap']}\n"
            f"- 每数据集重复 DATASET_REPEAT_RUNS={DATASET_REPEAT_RUNS}，批量外层 ITERATIONS={ITERATIONS}\n\n"
        )
        f.write(
            "| 数据集 | R1 | R2 | R3 | R4 | R5 | mean±std | 训练时长(min) | 开始时间 | 状态 |\n"
            "|--------|-----|-----|-----|-----|-----|----------|--------------|----------|------|\n"
        )


def append_batch_row(dataset_name, run_bests, mean_b, std_b, elapsed_min, start_time, status):
    # 表中展示名；兼容旧路径名 webstyle/subImages
    label = (
        'WebStyle'
        if dataset_name in ('webstyle', 'webstyle/subImages')
        else dataset_name
    )
    if len(run_bests) == DATASET_REPEAT_RUNS and status == 'OK':
        rpart = " | ".join(f"{x:.4f}" for x in run_bests)
        ms = f"{mean_b:.4f}±{std_b:.4f}"
    else:
        rpart = " | ".join(["—"] * DATASET_REPEAT_RUNS)
        ms = "—"
    with open(RESULT_FILE, 'a', encoding='utf-8') as f:
        f.write(
            f"| {label} | {rpart} | {ms} | {elapsed_min:.1f} | {start_time} | {status} |\n"
        )


def run_dataset(dataset_name, class_num, run_idx, total):
    """patch save（跳过 last）→ SSCtrain（含 dataset_repeat_runs）→ 返回值写入结果表 → 还原 save"""
    hp = _hparams_from_copy()
    safe_name    = dataset_name.replace('/', '_')
    current_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    model_name   = f'bat-{safe_name}-ep{hp["epochs"]}-lr{hp["base_lr"]}-ci{hp["cls_iter"]}'
    feature_name = f'{safe_name}_resnet50'
    log_file     = os.path.join(LOG_DIR, f'{model_name}-{current_time}.log')

    logger, fh, sh = setup_logger(log_file)
    logger.info('=' * 80)
    logger.info('[%d/%d] 开始训练: dataset=%s  class_num=%d',
                run_idx, total, dataset_name, class_num)
    logger.info(
        'hparams from parameter_load: epochs=%d, base_lr=%s, cls_iteration=%d, cls_lr=%s, '
        'cls_train_gap=%d, cls_test_gap=%d, image_size=%d, outer_iterations=%d, dataset_repeat_runs=%d',
        hp['epochs'], hp['base_lr'], hp['cls_iter'], hp['cls_lr'],
        hp['cls_train_gap'], hp['cls_test_gap'], hp['image_size'], ITERATIONS,
        DATASET_REPEAT_RUNS,
    )
    logger.info('=' * 80)

    orig_save = patch_skip_last_save()  # 只保存最佳模型

    data_source = os.path.join(DATA_ROOT, dataset_name) + '/'
    t_start     = time.time()
    start_time  = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    run_bests: list = []
    mean_b = std_b = 0.0
    status = 'OK'

    try:
        run_bests, mean_b, std_b = ssc_train_resnet.SSCtrain(
            logger, MODEL_PATH, current_time, model_name,
            data_source, class_num, ITERATIONS,
            TRAINING_MODE, BASE_MODEL_PATH,
            PRE_FEATURE_PATH, feature_name,
            train_args=None,
            dataset_repeat_runs=DATASET_REPEAT_RUNS,
        )
    except Exception as e:
        logger.error('训练异常: %s', str(e))
        status = f'ERROR: {e}'
        run_bests = []
        mean_b = std_b = 0.0
    finally:
        restore_save(orig_save)         # 还原 torch.save，不影响后续数据集
        logger.removeHandler(fh)
        logger.removeHandler(sh)
        fh.close()

    elapsed_min = (time.time() - t_start) / 60.0
    return run_bests, mean_b, std_b, elapsed_min, start_time, status


def main():
    write_batch_result_header()
    total = len(DATASETS)
    print(
        f"共 {total} 个数据集，每库 {DATASET_REPEAT_RUNS} 轮（仅记录每轮 best，汇总 mean±std）..."
    )

    for idx, (ds_name, cls_num) in enumerate(DATASETS, start=1):
        print(f"\n>>> [{idx}/{total}] 开始: {ds_name}")
        run_bests, mean_b, std_b, elapsed_min, start_time, status = run_dataset(
            ds_name, cls_num, idx, total
        )
        append_batch_row(ds_name, run_bests, mean_b, std_b, elapsed_min, start_time, status)
        if len(run_bests) == DATASET_REPEAT_RUNS and status == 'OK':
            print(
                f"<<< [{idx}/{total}] 完成: {ds_name}  mean±std={mean_b:.4f}±{std_b:.4f}  "
                f"{elapsed_min:.1f}min  [{status}]"
            )
        else:
            print(f"<<< [{idx}/{total}] 完成: {ds_name}  {elapsed_min:.1f}min  [{status}]")

    print(f"\n全部完成，结果已写入: {RESULT_FILE}")


if __name__ == '__main__':
    main()

