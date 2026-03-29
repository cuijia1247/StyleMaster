# ResNet50 SSC 超参数网格搜索脚本
# 通过 monkey-patch 覆盖 parameter_load() 来注入待测参数组合
# 结果汇总 CSV 存放于 ./param_optim/resnet50/
import sys
import os
import logging
import time
import csv
import itertools

# 将项目根目录加入 sys.path，以便直接 import ssc_train_resnet
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import ssc_train_resnet                                         # 导入主训练模块

# =====================================================================
# 参数搜索空间定义
# =====================================================================
SEARCH_SPACE = {
    'epochs':               list(range(50, 301, 50)),           # 50, 100, ..., 300
    'base_lr':              [round(v * 0.002, 4) for v in range(1, 6)],   # 0.002, 0.004, ..., 0.01
    'classifier_iteration': list(range(100, 301, 50)),          # 100, 150, ..., 300
}

# classifier_lr 固定使用主脚本默认值 0.0005
FIXED_CLASSIFIER_LR = 0.0005

# =====================================================================
# 固定配置（与主脚本保持一致）
# =====================================================================
DATASET_NAME    = 'Painting91'
CLASS_NUM_DICT  = {
    'Painting91': 13, 'Pandora': 12, 'WikiArt3': 15,
    'Arch': 25, 'FashionStyle14': 14, 'artbench': 10,
    'webstyle/subImages': 10, 'AVAstyle': 14
}
DATA_ROOT       = '/mnt/codes/data/style/'
MODEL_PATH      = os.path.join(ROOT, 'model') + '/'
PRE_FEATURE_PATH = os.path.join(ROOT, 'pretrainFeatures')
ITERATIONS      = 1                                             # 参数搜索时单次迭代即可，减少搜索时间
TRAINING_MODE   = 'original'
BASE_MODEL_PATH = '###'

RESULT_DIR      = os.path.join(ROOT, 'param_optim', 'resnet50')
os.makedirs(RESULT_DIR, exist_ok=True)

RESULT_CSV      = os.path.join(RESULT_DIR, 'grid_search_results.csv')
LOG_DIR         = os.path.join(RESULT_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)


def make_patched_parameter_load(epochs, base_lr, classifier_iteration, classifier_lr=FIXED_CLASSIFIER_LR):
    """
    生成一个覆盖 parameter_load() 的闭包，将指定超参数注入训练函数。
    其余参数沿用主脚本默认值。
    """
    def patched():
        backbone            = 'resnet50'
        ssc_backend         = 'resnet50'
        ssc_input           = 2048
        ssc_output          = 2048
        batch_size_         = 64
        batch_size_sample   = 'None'
        offset_bs           = 512
        image_size          = 64
        classifier_training_gap = 20
        classifier_test_gap     = 20
        model_name          = ''
        return (epochs, batch_size_, offset_bs, base_lr, image_size,
                classifier_iteration, classifier_lr, model_name, batch_size_sample,
                classifier_training_gap, backbone, ssc_backend, ssc_input, ssc_output,
                classifier_test_gap)
    return patched


def setup_logger(log_file: str) -> logging.Logger:
    """为每次参数组合创建独立的 logger，避免 handler 累积。"""
    logger_name = f"optim_{time.time()}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s - %(message)s")

    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger, fh, sh


def run_single(params: dict, run_idx: int, total: int) -> float:
    """
    执行单次超参数组合训练，返回 best_accuracy（float）。
    每次训练日志单独写入 LOG_DIR，不保存模型参数文件。
    """
    ep  = params['epochs']
    blr = params['base_lr']
    ci  = params['classifier_iteration']

    safe_name    = DATASET_NAME.replace('/', '_')
    current_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    model_name   = f'optim-{safe_name}-ep{ep}-lr{blr}-ci{ci}'
    feature_name = f'{safe_name}_resnet50'
    log_file     = os.path.join(LOG_DIR, f'{model_name}-{current_time}.log')

    logger, fh, sh = setup_logger(log_file)
    logger.info('=' * 80)
    logger.info('[%d/%d] 开始训练: epochs=%d, base_lr=%s, classifier_iteration=%d',
                run_idx, total, ep, blr, ci)
    logger.info('=' * 80)

    # monkey-patch parameter_load，将当前参数组合注入（classifier_lr 使用固定默认值）
    ssc_train_resnet.parameter_load = make_patched_parameter_load(ep, blr, ci)

    data_source  = os.path.join(DATA_ROOT, DATASET_NAME) + '/'
    class_number = CLASS_NUM_DICT.get(DATASET_NAME, 10)

    best_acc = 0.0
    try:
        ssc_train_resnet.SSCtrain(
            logger, MODEL_PATH, current_time, model_name,
            data_source, class_number, ITERATIONS,
            TRAINING_MODE, BASE_MODEL_PATH,
            PRE_FEATURE_PATH, feature_name
        )
        # 从日志末尾提取最佳精度
        # 格式: "... The best accuracy is 0.xxxx, and the last accuracy is 0.xxxx"
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                if 'best accuracy is' in line:
                    parts = line.strip().split('best accuracy is')
                    if len(parts) > 1:
                        best_acc = float(parts[1].split(',')[0].strip())
    except Exception as e:
        logger.error('训练异常: %s', str(e))
        best_acc = -1.0
    finally:
        logger.removeHandler(fh)
        logger.removeHandler(sh)
        fh.close()

    return best_acc


def main():
    # 生成全量参数组合
    keys   = list(SEARCH_SPACE.keys())
    values = list(SEARCH_SPACE.values())
    combos = list(itertools.product(*values))
    total  = len(combos)
    print(f'共 {total} 组参数组合，开始网格搜索...')

    # 准备 CSV 输出文件，只记录参数与 best_accuracy
    csv_fields = keys + ['best_accuracy']
    csv_exists = os.path.exists(RESULT_CSV)
    csv_file   = open(RESULT_CSV, 'a', newline='', encoding='utf-8')
    writer     = csv.DictWriter(csv_file, fieldnames=csv_fields)
    if not csv_exists:
        writer.writeheader()

    for idx, combo in enumerate(combos, start=1):
        params   = dict(zip(keys, combo))
        best_acc = run_single(params, idx, total)
        row      = {**params, 'best_accuracy': best_acc}
        writer.writerow(row)
        csv_file.flush()                                        # 每次写入后立即落盘，防止中途中断丢失结果
        print(f'[{idx}/{total}] 完成: {params} -> best_accuracy={best_acc:.4f}')

    csv_file.close()
    print(f'\n网格搜索完成，结果已保存至: {RESULT_CSV}')


if __name__ == '__main__':
    main()
