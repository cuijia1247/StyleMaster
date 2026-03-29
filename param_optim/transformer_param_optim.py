# Transformer SSC 超参数网格搜索脚本
# 通过 monkey-patch 覆盖 parameter_load() 来注入待测参数组合
# 结果汇总 CSV 存放于 ./param_optim/transformer/
import sys
import os
import logging
import time
import csv
import itertools
import shutil

# 将项目根目录加入 sys.path，以便直接 import ssc_train_transformer
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import ssc_train_transformer

# =====================================================================
# 参数搜索空间定义
# =====================================================================
SEARCH_SPACE = {
    "epochs": list(range(50, 301, 50)),  # 50, 100, ..., 300
    "base_lr": [round(v * 0.002, 4) for v in range(1, 6)],  # 0.002, 0.004, ..., 0.01
    "classifier_iteration": list(range(100, 301, 50)),  # 100, 150, ..., 300
}

# classifier_lr 固定使用 transformer 主脚本默认值
FIXED_CLASSIFIER_LR = 0.01

# =====================================================================
# 固定配置（与 transformer 主脚本保持一致）
# =====================================================================
DATASET_NAME = "Painting91"
CLASS_NUM_DICT = {
    "Painting91": 13,
    "Pandora": 12,
    "WikiArt3": 15,
    "WikiArt3_small": 15,
    "Arch": 25,
    "FashionStyle14": 14,
    "artbench": 10,
    "webstyle/subImages": 10,
    "AVAstyle": 14,
}
DATA_ROOT = "/mnt/codes/data/style/"
MODEL_PATH = os.path.join(ROOT, "model") + "/"
PRE_FEATURE_PATH = os.path.join(ROOT, "pretrainFeatures")
ITERATIONS = 1
TRAINING_MODE = "original"
BASE_MODEL_PATH = "###"

RESULT_DIR = os.path.join(ROOT, "param_optim", "transformer")
os.makedirs(RESULT_DIR, exist_ok=True)

RESULT_CSV = os.path.join(RESULT_DIR, "grid_search_results.csv")
LOG_DIR = os.path.join(RESULT_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
BEST_MODEL_DIR = os.path.join(RESULT_DIR, "best_models")
os.makedirs(BEST_MODEL_DIR, exist_ok=True)


def make_patched_parameter_load(
    epochs, base_lr, classifier_iteration, classifier_lr=FIXED_CLASSIFIER_LR
):
    """生成覆盖 parameter_load() 的闭包，将指定超参数注入训练函数。"""

    def patched():
        backbone = "vit_large_patch16_224"
        ssc_backend = "swin_base_patch4_window7_224"
        ssc_input = 1024
        ssc_output = 1024
        batch_size_ = 16
        batch_size_sample = "None"
        offset_bs = 512
        image_size = 224
        classifier_training_gap = 2  # 避免 epoch=0/1 触发分类器评估，明确不在 epoch=0 测试
        classifier_test_gap = 1
        model_name = ""
        return (
            epochs,
            batch_size_,
            offset_bs,
            base_lr,
            image_size,
            classifier_iteration,
            classifier_lr,
            model_name,
            batch_size_sample,
            classifier_training_gap,
            backbone,
            ssc_backend,
            ssc_input,
            ssc_output,
            classifier_test_gap,
        )

    return patched


def setup_logger(log_file: str):
    """为每次参数组合创建独立 logger，避免 handler 累积。"""
    logger_name = f"optim_transformer_{time.time()}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s - %(message)s")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger, fh, sh


def _pick_latest_by_suffix(paths, suffix):
    """从路径列表中选出指定后缀且最新生成的文件。"""
    candidates = [p for p in paths if p.endswith(suffix) and os.path.isfile(p)]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def _cleanup_model_files(paths):
    """删除给定模型文件列表，忽略不存在文件。"""
    removed = 0
    for path in paths:
        if os.path.isfile(path):
            try:
                os.remove(path)
                removed += 1
            except OSError:
                pass
    return removed


def _save_global_best_models(base_src, cls_src, params, best_acc):
    """保存全局最优模型（base + classifier）到 BEST_MODEL_DIR。"""
    for old_name in os.listdir(BEST_MODEL_DIR):
        old_path = os.path.join(BEST_MODEL_DIR, old_name)
        if os.path.isfile(old_path):
            os.remove(old_path)

    tag = (
        f"acc-{best_acc:.4f}-ep{params['epochs']}-lr{params['base_lr']}-"
        f"ci{params['classifier_iteration']}"
    )
    base_dst = os.path.join(BEST_MODEL_DIR, f"global-best-{tag}-SSC-base-best.pth")
    cls_dst = os.path.join(BEST_MODEL_DIR, f"global-best-{tag}-SSC-classifier-best.pth")
    shutil.copy2(base_src, base_dst)
    shutil.copy2(cls_src, cls_dst)
    return base_dst, cls_dst


def run_single(params: dict, run_idx: int, total: int):
    """执行单次超参数组合训练，返回精度与本次生成模型信息。"""
    ep = params["epochs"]
    blr = params["base_lr"]
    ci = params["classifier_iteration"]

    safe_name = DATASET_NAME.replace("/", "_")
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    model_name = f"optim-transformer-{safe_name}-ep{ep}-lr{blr}-ci{ci}"
    feature_name = f"{safe_name}_vit"
    log_file = os.path.join(LOG_DIR, f"{model_name}-{current_time}.log")

    logger, fh, sh = setup_logger(log_file)
    logger.info("=" * 80)
    logger.info(
        "[%d/%d] 开始训练: epochs=%d, base_lr=%s, classifier_iteration=%d",
        run_idx,
        total,
        ep,
        blr,
        ci,
    )
    logger.info("=" * 80)

    ssc_train_transformer.parameter_load = make_patched_parameter_load(ep, blr, ci)

    data_source = os.path.join(DATA_ROOT, DATASET_NAME) + "/"
    class_number = CLASS_NUM_DICT.get(DATASET_NAME, 10)

    best_acc = 0.0
    # 记录训练前模型目录状态，用于识别本次参数组合新生成的模型文件
    os.makedirs(MODEL_PATH, exist_ok=True)
    model_files_before = set(os.listdir(MODEL_PATH))
    try:
        ssc_train_transformer.SSCtrain(
            logger,
            MODEL_PATH,
            current_time,
            model_name,
            data_source,
            class_number,
            ITERATIONS,
            TRAINING_MODE,
            BASE_MODEL_PATH,
            PRE_FEATURE_PATH,
            feature_name,
        )

        # 从日志中提取最终 best accuracy
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                if "best accuracy is" in line:
                    parts = line.strip().split("best accuracy is")
                    if len(parts) > 1:
                        best_acc = float(parts[1].split(",")[0].strip())
    except Exception as e:
        logger.error("训练异常: %s", str(e))
        best_acc = -1.0
    finally:
        logger.removeHandler(fh)
        logger.removeHandler(sh)
        fh.close()

    model_files_after = set(os.listdir(MODEL_PATH))
    new_files = model_files_after - model_files_before
    current_new_model_paths = [
        os.path.join(MODEL_PATH, filename)
        for filename in new_files
        if filename.startswith(model_name)
    ]
    best_base_model = _pick_latest_by_suffix(current_new_model_paths, "-SSC-base-best.pth")
    best_cls_model = _pick_latest_by_suffix(
        current_new_model_paths, "-SSC-classifier-best.pth"
    )
    return best_acc, current_new_model_paths, best_base_model, best_cls_model


def main():
    keys = list(SEARCH_SPACE.keys())
    values = list(SEARCH_SPACE.values())
    combos = list(itertools.product(*values))
    total = len(combos)
    print(f"共 {total} 组参数组合，开始 Transformer 网格搜索...")

    csv_fields = keys + ["best_accuracy"]
    csv_exists = os.path.exists(RESULT_CSV)
    csv_file = open(RESULT_CSV, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    if not csv_exists:
        writer.writeheader()

    global_best_acc = -1.0
    global_best_saved = (None, None)

    for idx, combo in enumerate(combos, start=1):
        params = dict(zip(keys, combo))
        best_acc, run_model_files, run_base_best, run_cls_best = run_single(
            params, idx, total
        )

        # 仅保留全局最高准确率模型（base + classifier）
        if (
            best_acc > global_best_acc
            and run_base_best is not None
            and run_cls_best is not None
        ):
            global_best_saved = _save_global_best_models(
                run_base_best, run_cls_best, params, best_acc
            )
            global_best_acc = best_acc
            print(
                f"[{idx}/{total}] 刷新全局最优: best_accuracy={best_acc:.4f}, "
                f"模型已保存至 {BEST_MODEL_DIR}"
            )

        # 自动清理当前组合产生的所有模型文件（包括非最优组合及 last 模型）
        _cleanup_model_files(run_model_files)

        row = {**params, "best_accuracy": best_acc}
        writer.writerow(row)
        csv_file.flush()  # 每组训练后立即落盘，避免中断导致结果丢失
        print(f"[{idx}/{total}] 完成: {params} -> best_accuracy={best_acc:.4f}")

    csv_file.close()
    print(f"\n网格搜索完成，结果已保存至: {RESULT_CSV}")
    if global_best_saved[0] and global_best_saved[1]:
        print("全局最优模型保留如下:")
        print(f"  base: {global_best_saved[0]}")
        print(f"  cls : {global_best_saved[1]}")


if __name__ == "__main__":
    main()
