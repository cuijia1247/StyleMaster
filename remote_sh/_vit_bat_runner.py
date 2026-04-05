# ViT 批量训练 runner（由 run_ssc_train_vit_bat.sh 生成）
import sys
import os
import logging
import time
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import ssc_train_transformer

# ── 超参数（与 ssc_train_transformer.py parameter_load() 保持一致）──────────
# 分类器：四路融合 EfficientClassifier + Label Smoothing + 特征噪声扰动 + 早停(patience=5)
# SSC：余弦退火 lr，base_lr=0.001，image_size=112
EPOCHS            = 100
BASE_LR           = 0.001
CLS_ITERATION     = 80
CLASSIFIER_LR     = 0.0001
IMAGE_SIZE        = 112
CLS_TRAIN_GAP     = 10            # 每隔多少 epoch 触发一次分类器训练
CLS_TEST_GAP      = 2             # 分类器每训练多少轮评估一次测试集

# ── 固定配置 ─────────────────────────────────────────────────
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
PRE_FEATURE_PATH = os.path.join(ROOT, 'pretrainFeatures')
ITERATIONS       = 3
TRAINING_MODE    = 'original'
BASE_MODEL_PATH  = '###'

LOG_DIR     = os.path.join(ROOT, 'log')
RESULT_FILE = os.path.join(ROOT, 'remote_sh', 'vit_batch_result.md')

os.makedirs(LOG_DIR, exist_ok=True)


def make_patched_parameter_load(epochs, base_lr, classifier_iteration,
                                classifier_lr=CLASSIFIER_LR):
    """
    覆盖 ssc_train_transformer.parameter_load() 注入最优超参数
    """
    def patched():
        backbone                = 'vit_large_patch16_224'
        ssc_backend             = 'swin_base_patch4_window7_224'
        ssc_input               = 1024
        ssc_output              = 1024
        batch_size_             = 16        # Transformer 显存占用大，batch 较小
        batch_size_sample       = 'None'
        offset_bs               = 512
        image_size              = IMAGE_SIZE
        classifier_training_gap = CLS_TRAIN_GAP
        classifier_test_gap     = CLS_TEST_GAP
        model_name              = ''
        return (epochs, batch_size_, offset_bs, base_lr, image_size,
                classifier_iteration, classifier_lr, model_name, batch_size_sample,
                classifier_training_gap, backbone, ssc_backend, ssc_input, ssc_output,
                classifier_test_gap)
    return patched


def patch_skip_last_save():
    """
    替换 ssc_train_transformer 模块内的 torch.save，
    跳过文件名含 '-last.pth' 的末尾模型，只保留最佳模型（'-best.pth'）。
    """
    orig_save = torch.save

    def _save_best_only(obj, path, *args, **kwargs):
        if isinstance(path, str) and '-last.pth' in path:
            return
        orig_save(obj, path, *args, **kwargs)

    ssc_train_transformer.torch.save = _save_best_only
    return orig_save


def restore_save(orig_save):
    ssc_train_transformer.torch.save = orig_save


def setup_logger(log_file):
    """
    独立 logger，避免 handler 累积。
    命名格式与 ssc_train_transformer.py 一致
    """
    logger = logging.getLogger(f"vit_bat_{time.time()}")
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


def init_result_file():
    """初始化结果 Markdown 文件"""
    if not os.path.exists(RESULT_FILE):
        with open(RESULT_FILE, 'w', encoding='utf-8') as f:
            f.write("# ViT 批量训练结果\n\n")
            f.write(f"固定参数: epochs={EPOCHS}, base_lr={BASE_LR}(余弦退火), "
                    f"image_size={IMAGE_SIZE}, cls_iteration={CLS_ITERATION}, "
                    f"cls_lr={CLASSIFIER_LR}, cls_train_gap={CLS_TRAIN_GAP}, "
                    f"cls_test_gap={CLS_TEST_GAP}, iterations={ITERATIONS}, "
                    f"classifier=EfficientClassifier(四路融合+LabelSmoothing+FeatureNoise+ES5)\n\n")
            f.write("| 数据集 | 最佳准确率 | 最终准确率 | 训练时长(min) | 开始时间 | 状态 |\n")
            f.write("|--------|-----------|-----------|--------------|----------|------|\n")
    else:
        with open(RESULT_FILE, 'a', encoding='utf-8') as f:
            f.write(f"\n---\n_续跑于 {time.strftime('%Y-%m-%d %H:%M:%S')} "
                    f"| epochs={EPOCHS}, base_lr={BASE_LR}, image_size={IMAGE_SIZE}, "
                    f"cls_iter={CLS_ITERATION}, cls_lr={CLASSIFIER_LR}, iter={ITERATIONS}_\n\n")
            f.write("| 数据集 | 最佳准确率 | 最终准确率 | 训练时长(min) | 开始时间 | 状态 |\n")
            f.write("|--------|-----------|-----------|--------------|----------|------|\n")


def append_result(dataset_name, best_acc, last_acc, elapsed_min, start_time, status):
    label = dataset_name.replace('webstyle/subImages', 'WebStyle')
    with open(RESULT_FILE, 'a', encoding='utf-8') as f:
        f.write(f"| {label} | {best_acc:.4f} | {last_acc:.4f} "
                f"| {elapsed_min:.1f} | {start_time} | {status} |\n")


def parse_accuracy(log_file):
    """
    从日志末行解析精度，格式：
      'The best accuracy is 0.xxxx, and the last accuracy is 0.xxxx'
    """
    best_acc = last_acc = 0.0
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            if 'best accuracy is' in line and 'last accuracy is' in line:
                try:
                    seg      = line.split('best accuracy is')[1]
                    best_acc = float(seg.split(',')[0].strip())
                    last_acc = float(seg.split('last accuracy is')[1].strip())
                except (IndexError, ValueError):
                    pass
    return best_acc, last_acc


def run_dataset(dataset_name, class_num, run_idx, total):
    """
    单数据集训练入口
    """
    safe_name    = dataset_name.replace('/', '_')
    current_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    model_name   = f'vit-bat-{safe_name}-ep{EPOCHS}-lr{BASE_LR}-ci{CLS_ITERATION}'
    feature_name = f'{safe_name}_vit'          # ViT 特征后缀
    log_file     = os.path.join(LOG_DIR, f'{model_name}-{current_time}.log')

    logger, fh, sh = setup_logger(log_file)
    logger.info('=' * 80)
    logger.info('[%d/%d] 开始训练: dataset=%s  class_num=%d',
                run_idx, total, dataset_name, class_num)
    logger.info('epochs=%d, base_lr=%s, cls_iteration=%d, '
                'cls_train_gap=%d, cls_test_gap=%d',
                EPOCHS, BASE_LR, CLS_ITERATION, CLS_TRAIN_GAP, CLS_TEST_GAP)
    logger.info('=' * 80)

    # monkey-patch parameter_load
    ssc_train_transformer.parameter_load = make_patched_parameter_load(
        EPOCHS, BASE_LR, CLS_ITERATION)
    orig_save = patch_skip_last_save()

    data_source = os.path.join(DATA_ROOT, dataset_name) + '/'
    t_start     = time.time()
    start_time  = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    best_acc    = last_acc = 0.0
    status      = 'OK'

    try:
        ssc_train_transformer.SSCtrain(
            logger, MODEL_PATH, current_time, model_name,
            data_source, class_num, ITERATIONS,
            TRAINING_MODE, BASE_MODEL_PATH,
            PRE_FEATURE_PATH, feature_name
        )
        best_acc, last_acc = parse_accuracy(log_file)
    except Exception as e:
        logger.error('训练异常: %s', str(e))
        status = f'ERROR: {e}'
        best_acc = last_acc = -1.0
    finally:
        restore_save(orig_save)
        logger.removeHandler(fh)
        logger.removeHandler(sh)
        fh.close()

    elapsed_min = (time.time() - t_start) / 60.0
    return best_acc, last_acc, elapsed_min, start_time, status


def main():
    init_result_file()
    total = len(DATASETS)
    print(f"共 {total} 个数据集，依次开始批量训练...")

    for idx, (ds_name, cls_num) in enumerate(DATASETS, start=1):
        print(f"\n>>> [{idx}/{total}] 开始: {ds_name}")
        best_acc, last_acc, elapsed_min, start_time, status = \
            run_dataset(ds_name, cls_num, idx, total)
        append_result(ds_name, best_acc, last_acc, elapsed_min, start_time, status)
        print(f"<<< [{idx}/{total}] 完成: {ds_name}  "
              f"best={best_acc:.4f}  last={last_acc:.4f}  "
              f"{elapsed_min:.1f}min  [{status}]")

    print(f"\n全部完成，结果已写入: {RESULT_FILE}")


if __name__ == '__main__':
    main()
