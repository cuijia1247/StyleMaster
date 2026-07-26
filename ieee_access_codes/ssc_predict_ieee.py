#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SSC-ResNet50 测试集推理与错误案例收集。

对 ieee_access_paperdata/models/ 下各数据集最佳 checkpoint 在 test 集上推理，
将预测错误样本写入 ieee_access_paperdata/ssc_failure_case_list.md。

用法（项目根目录）::
  python ieee_access_codes/ssc_predict_ieee.py
  python ieee_access_codes/ssc_predict_ieee.py --datasets Painting91 Pandora
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from SscDataSet_new import SscDataset  # noqa: E402
from ssc.utils import MultiViewDataInjector, get_ssc_transforms  # noqa: E402
from ssc_train_resnet_copy import parameter_load  # noqa: E402
from utils.pretrainFeatureExtraction import load_dataFeatures  # noqa: E402

DEFAULT_DATA_BASE = "/mnt/codes/data/style"
DEFAULT_MODEL_DIR = os.path.join(_ROOT, "ieee_access_paperdata", "models")
DEFAULT_PRE_FEATURE = os.path.join(_ROOT, "pretrainFeatures")
DEFAULT_OUT_MD = os.path.join(_ROOT, "ieee_access_paperdata", "ssc_failure_case_list.md")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 各数据集推理配置：base + classifier 成对 checkpoint
DATASET_JOBS = [
    {
        "name": "Painting91",
        "rel_dir": "Painting91",
        "num_classes": 13,
        "feature_name": "Painting91_resnet50",
        "base_path": os.path.join(
            DEFAULT_MODEL_DIR,
            "ssc-Painting91-SSC-resnet50-2026-07-03-09-32-06-run2-"
            "iteration-0-accuracy-7353-SSC-base-best.pth",
        ),
        "classifier_path": os.path.join(
            DEFAULT_MODEL_DIR,
            "ssc-Painting91-SSC-resnet50-2026-07-03-09-32-06-run2-"
            "iteration-0-accuracy-7353-SSC-classifier-best.pth",
        ),
    },
    {
        "name": "Pandora",
        "rel_dir": "Pandora",
        "num_classes": 12,
        "feature_name": "Pandora_resnet50",
        "base_path": os.path.join(
            DEFAULT_MODEL_DIR,
            "ssc-Pandora-SSC-resnet50-2026-07-03-16-43-17-run0-"
            "iteration-0-accuracy-6153-SSC-base-best.pth",
        ),
        "classifier_path": os.path.join(
            DEFAULT_MODEL_DIR,
            "ssc-Pandora-SSC-resnet50-2026-07-03-16-43-17-run0-"
            "iteration-0-accuracy-6153-SSC-classifier-best.pth",
        ),
    },
    {
        "name": "FashionStyle14",
        "rel_dir": "FashionStyle14",
        "num_classes": 14,
        "feature_name": "FashionStyle14_resnet50",
        "base_path": os.path.join(
            DEFAULT_MODEL_DIR,
            "ssc-FashionStyle14-SSC-resnet50-2026-07-03-23-15-33-run2-"
            "iteration-0-accuracy-7015-SSC-base-best.pth",
        ),
        "classifier_path": os.path.join(
            DEFAULT_MODEL_DIR,
            "ssc-FashionStyle14-SSC-resnet50-2026-07-03-23-15-33-run2-"
            "iteration-0-accuracy-7015-SSC-classifier-best.pth",
        ),
    },
    {
        "name": "Arch",
        "rel_dir": "Arch",
        "num_classes": 25,
        "feature_name": "Arch_resnet50",
        "base_path": os.path.join(
            DEFAULT_MODEL_DIR,
            "ssc-Arch-SSC-resnet50-2026-07-04-03-45-21-run2-"
            "iteration-0-accuracy-6829-SSC-base-best.pth",
        ),
        "classifier_path": os.path.join(
            DEFAULT_MODEL_DIR,
            "ssc-Arch-SSC-resnet50-2026-07-04-03-45-21-run2-"
            "iteration-0-accuracy-6829-SSC-classifier-best.pth",
        ),
    },
]

@dataclass
class FailureCase:
    """单条预测错误记录。"""
    file_path: str
    true_label: int
    pred_label: int
    true_name: str
    pred_name: str


@dataclass
class DatasetPredictResult:
    """单数据集推理汇总。"""
    name: str
    num_classes: int
    data_root: str
    base_path: str
    classifier_path: str
    accuracy: float
    total: int
    num_errors: int
    inference_time_s: float
    failures: List[FailureCase]


def infer_class_names(data_root: str, num_classes: int) -> List[str]:
    """从 test/ 各类别首样本文件名推断可读类别名。"""
    names: List[str] = []
    test_root = os.path.join(data_root, "test")
    for cls_id in range(1, num_classes + 1):
        folder = os.path.join(test_root, str(cls_id))
        if not os.path.isdir(folder):
            names.append(f"class_{cls_id}")
            continue
        sample = sorted(os.listdir(folder))[0]
        stem = os.path.splitext(sample)[0]
        stem = re.sub(r"_\d+$", "", stem)
        names.append(stem.replace("_", " "))
    return names


def _sample_path(data_root: str, label_1based: int, filename: str) -> str:
    return os.path.abspath(os.path.join(data_root, "test", str(label_1based), filename))


@torch.no_grad()
def predict_test_set_with_failures(
    model: torch.nn.Module,
    classifier: torch.nn.Module,
    data_root: str,
    feature_dict: dict,
    class_names: List[str],
    image_size: int,
    batch_size: int,
) -> DatasetPredictResult:
    """在 test 集上推理，返回准确率与错误案例列表。"""
    data_source = data_root.rstrip("/") + "/"
    _, _, transform_eval = get_ssc_transforms(
        image_size, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    )
    testset = SscDataset(
        data_source,
        "test",
        transform=MultiViewDataInjector([transform_eval, transform_eval]),
    )
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    model.eval()
    classifier.eval()

    y_true: List[int] = []
    y_pred: List[int] = []
    failures: List[FailureCase] = []

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_infer_start = time.perf_counter()

    for view1, view2, label, names, _ in testloader:
        view1 = view1.to(device)
        view2 = view2.to(device)
        backbone_view = torch.stack([feature_dict[n] for n in names], dim=0).to(device)
        ssc_v1 = model(view1)
        ssc_v2 = model(view2)
        logits = classifier(ssc_v1, ssc_v2, backbone_view)
        pred = logits.argmax(dim=1)

        labels_0 = (label - 1).long().cpu().tolist()
        preds_0 = pred.cpu().tolist()
        y_true.extend(labels_0)
        y_pred.extend(preds_0)

        for fname, true_0, pred_0, label_1 in zip(
            names, labels_0, preds_0, label.tolist()
        ):
            if true_0 != pred_0:
                failures.append(
                    FailureCase(
                        file_path=_sample_path(data_root.rstrip("/"), label_1, fname),
                        true_label=true_0,
                        pred_label=pred_0,
                        true_name=class_names[true_0],
                        pred_name=class_names[pred_0],
                    )
                )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_time_s = time.perf_counter() - t_infer_start

    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)
    accuracy = float(np.mean(y_true_arr == y_pred_arr))
    return DatasetPredictResult(
        name="",
        num_classes=len(class_names),
        data_root=data_root.rstrip("/"),
        base_path="",
        classifier_path="",
        accuracy=accuracy,
        total=len(y_true_arr),
        num_errors=len(failures),
        inference_time_s=inference_time_s,
        failures=failures,
    )


def run_dataset_job(
    job: dict,
    data_base: str,
    pre_feature_path: str,
    batch_size: int,
    image_size: int,
    logger: logging.Logger,
) -> DatasetPredictResult:
    """加载指定 checkpoint 并在 test 集上推理。"""
    base_path = job["base_path"]
    classifier_path = job["classifier_path"]
    for path in (base_path, classifier_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"模型文件不存在: {path}")

    data_root = os.path.join(data_base.rstrip("/"), job["rel_dir"])
    feature_path = os.path.join(
        pre_feature_path, f"{job['feature_name']}_test_features.pkl"
    )
    if not os.path.isfile(feature_path):
        raise FileNotFoundError(f"预提取特征不存在: {feature_path}")

    logger.info("[%s] 加载模型", job["name"])
    logger.info("  base:       %s", base_path)
    logger.info("  classifier: %s", classifier_path)
    logger.info("  data_root:  %s", data_root)

    model = torch.load(base_path, map_location=device).to(device).eval()
    classifier = torch.load(classifier_path, map_location=device).to(device).eval()
    feature_dict = load_dataFeatures(feature_path)
    class_names = infer_class_names(data_root, job["num_classes"])

    result = predict_test_set_with_failures(
        model, classifier, data_root, feature_dict, class_names, image_size, batch_size
    )
    result.name = job["name"]
    result.base_path = base_path
    result.classifier_path = classifier_path

    logger.info(
        "[%s] test Accuracy=%.4f, errors=%d/%d, inference_time=%.3fs",
        job["name"],
        result.accuracy,
        result.num_errors,
        result.total,
        result.inference_time_s,
    )
    return result


def write_failure_markdown(
    out_path: str,
    results: List[DatasetPredictResult],
    argv_summary: str,
) -> None:
    """将各数据集错误案例写入 Markdown。"""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# SSC-ResNet50 测试集预测错误案例",
        "",
        f"_生成时间: {timestamp}_",
        "",
        f"_命令: `{argv_summary}`_",
        "",
        "## 汇总",
        "",
        "| Dataset | num_classes | Test Acc | 推理时间(s) | 错误数 | 测试样本数 | base 模型 | classifier 模型 |",
        "|---------|-------------|----------|------------|--------|-----------|-----------|----------------|",
    ]

    for r in results:
        lines.append(
            f"| {r.name} | {r.num_classes} | {r.accuracy:.4f} | {r.inference_time_s:.3f} "
            f"| {r.num_errors} | {r.total} "
            f"| `{r.base_path}` | `{r.classifier_path}` |"
        )
    lines.append("")

    for r in results:
        lines.extend([
            f"## {r.name}",
            "",
            f"- **data_root**: `{r.data_root}`",
            f"- **Test Accuracy**: {r.accuracy:.4f}",
            f"- **推理时间**: {r.inference_time_s:.3f} s",
            f"- **错误数 / 总数**: {r.num_errors} / {r.total}",
            "",
            "| # | 文件路径 | 真实类别 (id) | 预测类别 (id) |",
            "|---|---------|--------------|--------------|",
        ])
        for idx, case in enumerate(r.failures, start=1):
            lines.append(
                f"| {idx} | `{case.file_path}` | {case.true_name} ({case.true_label}) "
                f"| {case.pred_name} ({case.pred_label}) |"
            )
        if not r.failures:
            lines.append("| - | _无错误样本_ | - | - |")
        lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SSC-ResNet50 测试集推理与错误案例收集")
    p.add_argument(
        "--datasets",
        nargs="+",
        default=[j["name"] for j in DATASET_JOBS],
        choices=[j["name"] for j in DATASET_JOBS],
    )
    p.add_argument("--data_base", type=str, default=DEFAULT_DATA_BASE)
    p.add_argument("--pre_feature_path", type=str, default=DEFAULT_PRE_FEATURE)
    p.add_argument("--out_md", type=str, default=DEFAULT_OUT_MD)
    return p.parse_args()


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("ssc_predict_ieee")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger
    fmt = logging.Formatter("%(asctime)s - %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    os.makedirs(os.path.join(_ROOT, "log"), exist_ok=True)
    log_path = os.path.join(
        _ROOT, "log", f"ssc_predict_ieee_{time.strftime('%Y%m%d_%H%M%S')}.log"
    )
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def main() -> None:
    args = parse_args()
    os.chdir(_ROOT)
    logger = setup_logger()

    params = parameter_load()
    _, batch_size, _, _, image_size, *_ = params

    job_map = {j["name"]: j for j in DATASET_JOBS}
    selected = [job_map[name] for name in args.datasets]

    logger.info("Device: %s", device)
    logger.info("Datasets: %s", ", ".join(args.datasets))

    results: List[DatasetPredictResult] = []
    for job in selected:
        logger.info("=" * 80)
        results.append(
            run_dataset_job(
                job,
                args.data_base,
                args.pre_feature_path,
                batch_size,
                image_size,
                logger,
            )
        )

    argv_summary = " ".join([sys.argv[0]] + sys.argv[1:])
    write_failure_markdown(args.out_md, results, argv_summary)

    print(f"\n错误案例已写入: {args.out_md}")
    total_infer_s = 0.0
    for r in results:
        total_infer_s += r.inference_time_s
        print(
            f"  [{r.name}] acc={r.accuracy:.4f}, "
            f"errors={r.num_errors}/{r.total}, "
            f"inference_time={r.inference_time_s:.3f}s"
        )
    if len(results) > 1:
        print(f"  [合计] inference_time={total_infer_s:.3f}s")


if __name__ == "__main__":
    main()
