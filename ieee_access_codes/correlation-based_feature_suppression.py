#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Correlation-based Feature Suppression 实验脚本。

加载已训练 SSC-base（冻结），分别训练五种 IEEE 分类头并评估：
  1. EfficientClassifier      — 全通道软正交抑制
  2. NoSuppressClassifier     — 不抑制，仅用 backbone
  3. RandomSuppressClassifier — 随机 20% 通道抑制
  4. LowCorClassifier         — 低相关 20% 通道抑制
  5. HighCorClassifier        — 高相关 20% 通道抑制

用法（项目根目录）::
  python ieee_access_codes/correlation-based_feature_suppression.py \\
    --ssc_base_path model/ssc-Painting91-...-SSC-base-best.pth \\
    --dataset_name Painting91 --data_root /mnt/codes/data/style/
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

DEFAULT_SSC_BASE = os.path.join(
    _ROOT,
    "ieee_access_paperdata/models/"
    "ssc-Painting91-SSC-resnet50-2026-07-03-09-32-06-run2-iteration-0-accuracy-7353-SSC-base-best.pth",
)
DEFAULT_MODEL_OUT = os.path.join(
    _ROOT, "ieee_access_paperdata/models/new_models_for_ieee"
)
DEFAULT_RESULT_MD = os.path.join(
    _ROOT, "ieee_access_paperdata/correlation-based_feature_suppression.md"
)

# 三数据集 benchmark 对应的 SSC-base（--datasets 多数据集模式默认使用）
BENCHMARK_SSC_BASE: Dict[str, str] = {
    "Painting91": os.path.join(
        _ROOT,
        "ieee_access_paperdata/models/"
        "ssc-Painting91-SSC-resnet50-2026-07-03-09-32-06-run2-iteration-0-accuracy-7353-SSC-base-best.pth",
    ),
    "FashionStyle14": os.path.join(
        _ROOT,
        "ieee_access_paperdata/models/"
        "ssc-FashionStyle14-SSC-resnet50-2026-07-03-23-15-33-run2-iteration-0-accuracy-7015-SSC-base-best.pth",
    ),
    "Arch": os.path.join(
        _ROOT,
        "ieee_access_paperdata/models/"
        "ssc-Arch-SSC-resnet50-2026-07-04-03-45-21-run2-iteration-0-accuracy-6829-SSC-base-best.pth",
    ),
}

METRIC_KEYS = ("accuracy", "macro_f1", "weighted_f1", "balanced_accuracy")

# 分类器保存文件名前缀
MODE_SAVE_PREFIX: Dict[str, str] = {
    "full": "ssc",
    "none": "ssc-NoSuppressed",
    "random20": "ssc-RandomSuppressed",
    "lowcor": "ssc-LowCorSuppressed",
    "highcor": "ssc-HighCorSuppressed",
}

from SscDataSet_new import SscDataset  # noqa: E402
from ssc.classifier_ieee import (  # noqa: E402
    EfficientClassifier,
    HighCorClassifier,
    LowCorClassifier,
    NoSuppressClassifier,
    RandomSuppressClassifier,
)
from ssc.utils import MultiViewDataInjector, get_ssc_transforms  # noqa: E402
from ssc_train_resnet_copy import merge_params_with_args, parameter_load  # noqa: E402
from utils.pretrainFeatureExtraction import load_dataFeatures  # noqa: E402

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CLASS_NUM_DICT: Dict[str, int] = {
    "Painting91": 13,
    "Pandora": 12,
    "WikiArt3": 15,
    "Arch": 25,
    "FashionStyle14": 14,
    "Artbench": 10,
    "artbench": 10,
    "webstyle": 10,
    "AVAstyle": 14,
}

CacheBatch = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass
class TrainResult:
    classifier_name: str
    best_test_accuracy: float
    classifier_path: str
    metrics: Dict[str, float]
    run_idx: int = 0


@dataclass
class AggregatedResult:
    classifier_name: str
    mean_metrics: Dict[str, float]
    std_metrics: Dict[str, float]
    run_results: List[TrainResult]


def set_run_seed(seed: int) -> None:
    """固定单次 repeat 的随机性（分类器初始化、RandomSuppress、训练噪声）。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def aggregate_run_results(all_runs: List[List[TrainResult]]) -> List[AggregatedResult]:
    """按分类器聚合多次 run，计算各指标的 mean / std。"""
    if not all_runs:
        return []
    classifier_names = [r.classifier_name for r in all_runs[0]]
    aggregated: List[AggregatedResult] = []
    for name in classifier_names:
        per_run = [
            next(r for r in run_results if r.classifier_name == name)
            for run_results in all_runs
        ]
        mean_metrics: Dict[str, float] = {}
        std_metrics: Dict[str, float] = {}
        for key in METRIC_KEYS:
            values = np.asarray([r.metrics.get(key, 0.0) for r in per_run], dtype=np.float64)
            mean_metrics[key] = float(values.mean())
            std_metrics[key] = float(values.std(ddof=0))
        aggregated.append(
            AggregatedResult(
                classifier_name=name,
                mean_metrics=mean_metrics,
                std_metrics=std_metrics,
                run_results=per_run,
            )
        )
    return aggregated


def _make_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger(log_path)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s - %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def load_frozen_ssc_base(base_path: str) -> nn.Module:
    """加载 SSC-base 并冻结全部参数（参考 ssc_predict.py）。"""
    if not os.path.isfile(base_path):
        raise FileNotFoundError(f"SSC-base 不存在: {base_path}")
    model = torch.load(base_path, map_location=device)
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def build_ssc_caches(
    model: nn.Module,
    data_source: str,
    train_feature_dict: dict,
    test_feature_dict: dict,
    image_size: int,
    batch_size: int,
    cache_rounds: int = 12,
) -> Tuple[List[List[CacheBatch]], List[CacheBatch], int, int]:
    """构建 train K 份缓存 + test 缓存（逻辑同 ssc_train_resnet_copy 分类器段）。"""
    transform_t, transform_t1, _ = get_ssc_transforms(
        image_size, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    )
    trainset = SscDataset(
        data_source, "train", transform=MultiViewDataInjector([transform_t, transform_t1])
    )
    testset = SscDataset(
        data_source, "test", transform=MultiViewDataInjector([transform_t, transform_t1])
    )
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    train_caches: List[List[CacheBatch]] = []
    with torch.no_grad():
        for _ in range(cache_rounds):
            cache: List[CacheBatch] = []
            for view1, view2, label, names, _ in trainloader:
                view1 = view1.to(device)
                view2 = view2.to(device)
                bb = torch.stack([train_feature_dict[n] for n in names], dim=0).to(device)
                ssc_v1 = model(view1)
                ssc_v2 = model(view2)
                cache.append(
                    (bb.cpu(), ssc_v1.cpu(), ssc_v2.cpu(), (label - 1).long().cpu())
                )
            train_caches.append(cache)

        test_cache: List[CacheBatch] = []
        for view1, view2, label, names, _ in testloader:
            view1 = view1.to(device)
            view2 = view2.to(device)
            bb = torch.stack([test_feature_dict[n] for n in names], dim=0).to(device)
            ssc_v1 = model(view1)
            ssc_v2 = model(view2)
            test_cache.append(
                (bb.cpu(), ssc_v1.cpu(), ssc_v2.cpu(), (label - 1).long().cpu())
            )

    return train_caches, test_cache, len(trainset), len(testset)


@torch.no_grad()
def evaluate_classifier(
    classifier: nn.Module,
    test_cache: List[CacheBatch],
    class_number: int,
) -> Dict[str, float]:
    classifier.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    for bb_feat, ssc_v1, ssc_v2, label in test_cache:
        bb_feat = bb_feat.to(device)
        ssc_v1 = ssc_v1.to(device)
        ssc_v2 = ssc_v2.to(device)
        logits = classifier(ssc_v1, ssc_v2, bb_feat)
        pred = logits.argmax(dim=1)
        y_true.extend(label.tolist())
        y_pred.extend(pred.cpu().tolist())
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)
    labels_all = list(range(class_number))
    return {
        "accuracy": float(np.mean(y_true_arr == y_pred_arr)),
        "macro_f1": float(
            f1_score(y_true_arr, y_pred_arr, average="macro", labels=labels_all, zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true_arr, y_pred_arr, average="weighted", labels=labels_all, zero_division=0)
        ),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred_arr)),
    }


def _train_classifier_loop(
    classifier: nn.Module,
    train_caches: List[List[CacheBatch]],
    test_cache: List[CacheBatch],
    train_size: int,
    classifier_iterations: int,
    classifier_lr: float,
    classifier_test_gap: int,
    class_number: int,
    logger: logging.Logger,
) -> Tuple[float, Dict[str, float], nn.Module]:
    """分类器训练主循环（SSC-base 已冻结，仅更新 classifier）。"""
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=classifier_lr, weight_decay=1e-3)
    best_acc = 0.0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_metrics: Dict[str, float] = {}

    for i in range(classifier_iterations):
        classifier.train()
        rot_cache = train_caches[i % len(train_caches)]
        total_correct = 0.0

        for bb_feat, ssc_v1, ssc_v2, label in rot_cache:
            bb_feat = bb_feat.to(device)
            ssc_v1 = ssc_v1.to(device)
            ssc_v2 = ssc_v2.to(device)
            label = label.to(device)
            bb_feat = bb_feat + torch.randn_like(bb_feat) * 0.01
            ssc_v1 = ssc_v1 + torch.randn_like(ssc_v1) * 0.01
            ssc_v2 = ssc_v2 + torch.randn_like(ssc_v2) * 0.01

            logits = classifier(ssc_v1, ssc_v2, bb_feat)
            loss = criterion(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pred = logits.argmax(dim=1)
                total_correct += pred.eq(label).sum().item()

        if (i + 1) % max(1, classifier_test_gap) == 0 or i == classifier_iterations - 1:
            metrics = evaluate_classifier(classifier, test_cache, class_number)
            logger.info(
                "  iter %d/%d train_acc=%.4f test_acc=%.4f macro_f1=%.4f",
                i + 1,
                classifier_iterations,
                total_correct / train_size,
                metrics["accuracy"],
                metrics["macro_f1"],
            )
            if metrics["accuracy"] > best_acc:
                best_acc = metrics["accuracy"]
                best_metrics = metrics
                best_state = {k: v.detach().cpu().clone() for k, v in classifier.state_dict().items()}

    if best_state is not None:
        classifier.load_state_dict(best_state)
    return best_acc, best_metrics, classifier


def _run_classifier_pipeline(
    classifier_cls: Type[nn.Module],
    classifier_name: str,
    model: nn.Module,
    train_caches: List[List[CacheBatch]],
    test_cache: List[CacheBatch],
    train_size: int,
    ssc_output: int,
    class_number: int,
    classifier_iterations: int,
    classifier_lr: float,
    classifier_test_gap: int,
    model_save_dir: str,
    save_prefix: str,
    safe_dataset: str,
    time_str: str,
    logger: logging.Logger,
    run_idx: int = 0,
    repeat_runs: int = 1,
    **classifier_kwargs,
) -> TrainResult:
    logger.info("========== 训练 %s (run %d) ==========", classifier_name, run_idx + 1)
    classifier = classifier_cls(ssc_output, class_number, **classifier_kwargs).to(device)
    best_acc, metrics, classifier = _train_classifier_loop(
        classifier,
        train_caches,
        test_cache,
        train_size,
        classifier_iterations,
        classifier_lr,
        classifier_test_gap,
        class_number,
        logger,
    )
    acc_str = f"{best_acc:.4f}".split(".")[1][:4]
    run_suffix = f"-run{run_idx + 1}" if repeat_runs > 1 else ""
    save_name = (
        f"{save_prefix}-{safe_dataset}-{time_str}{run_suffix}-iteration-0-"
        f"accuracy-{acc_str}-SSC-classifier-best.pth"
    )
    save_path = os.path.join(model_save_dir, save_name)
    torch.save(classifier, save_path)
    logger.info("[%s] best_test_acc=%.4f saved=%s", classifier_name, best_acc, save_path)
    return TrainResult(
        classifier_name=classifier_name,
        best_test_accuracy=best_acc,
        classifier_path=save_path,
        metrics=metrics,
        run_idx=run_idx,
    )


def train_full_suppress_classifier(
    model: nn.Module,
    train_caches: List[List[CacheBatch]],
    test_cache: List[CacheBatch],
    train_size: int,
    ssc_output: int,
    class_number: int,
    classifier_iterations: int,
    classifier_lr: float,
    classifier_test_gap: int,
    model_save_dir: str,
    save_prefix: str,
    safe_dataset: str,
    time_str: str,
    logger: logging.Logger,
    **kwargs,
) -> TrainResult:
    """流程 1：EfficientClassifier 全通道软正交抑制。"""
    return _run_classifier_pipeline(
        EfficientClassifier,
        "EfficientClassifier",
        model,
        train_caches,
        test_cache,
        train_size,
        ssc_output,
        class_number,
        classifier_iterations,
        classifier_lr,
        classifier_test_gap,
        model_save_dir,
        save_prefix,
        safe_dataset,
        time_str,
        logger,
        **kwargs,
    )


def train_no_suppress_classifier(
    model: nn.Module,
    train_caches: List[List[CacheBatch]],
    test_cache: List[CacheBatch],
    train_size: int,
    ssc_output: int,
    class_number: int,
    classifier_iterations: int,
    classifier_lr: float,
    classifier_test_gap: int,
    model_save_dir: str,
    save_prefix: str,
    safe_dataset: str,
    time_str: str,
    logger: logging.Logger,
    **kwargs,
) -> TrainResult:
    """流程 2：NoSuppressClassifier，仅 backbone 分类。"""
    return _run_classifier_pipeline(
        NoSuppressClassifier,
        "NoSuppressClassifier",
        model,
        train_caches,
        test_cache,
        train_size,
        ssc_output,
        class_number,
        classifier_iterations,
        classifier_lr,
        classifier_test_gap,
        model_save_dir,
        save_prefix,
        safe_dataset,
        time_str,
        logger,
        **kwargs,
    )


def train_random_suppress_classifier(
    model: nn.Module,
    train_caches: List[List[CacheBatch]],
    test_cache: List[CacheBatch],
    train_size: int,
    ssc_output: int,
    class_number: int,
    classifier_iterations: int,
    classifier_lr: float,
    classifier_test_gap: int,
    model_save_dir: str,
    save_prefix: str,
    safe_dataset: str,
    time_str: str,
    logger: logging.Logger,
    suppress_ratio: float = 0.2,
    **kwargs,
) -> TrainResult:
    """流程 3：RandomSuppressClassifier，随机 20%% 通道软正交抑制。"""
    return _run_classifier_pipeline(
        RandomSuppressClassifier,
        "RandomSuppressClassifier",
        model,
        train_caches,
        test_cache,
        train_size,
        ssc_output,
        class_number,
        classifier_iterations,
        classifier_lr,
        classifier_test_gap,
        model_save_dir,
        save_prefix,
        safe_dataset,
        time_str,
        logger,
        suppress_ratio=suppress_ratio,
        **kwargs,
    )


def train_low_cor_classifier(
    model: nn.Module,
    train_caches: List[List[CacheBatch]],
    test_cache: List[CacheBatch],
    train_size: int,
    ssc_output: int,
    class_number: int,
    classifier_iterations: int,
    classifier_lr: float,
    classifier_test_gap: int,
    model_save_dir: str,
    save_prefix: str,
    safe_dataset: str,
    time_str: str,
    logger: logging.Logger,
    suppress_ratio: float = 0.2,
    **kwargs,
) -> TrainResult:
    """流程 4：LowCorClassifier，低相关 20%% 通道软正交抑制。"""
    return _run_classifier_pipeline(
        LowCorClassifier,
        "LowCorClassifier",
        model,
        train_caches,
        test_cache,
        train_size,
        ssc_output,
        class_number,
        classifier_iterations,
        classifier_lr,
        classifier_test_gap,
        model_save_dir,
        save_prefix,
        safe_dataset,
        time_str,
        logger,
        suppress_ratio=suppress_ratio,
        **kwargs,
    )


def train_high_cor_classifier(
    model: nn.Module,
    train_caches: List[List[CacheBatch]],
    test_cache: List[CacheBatch],
    train_size: int,
    ssc_output: int,
    class_number: int,
    classifier_iterations: int,
    classifier_lr: float,
    classifier_test_gap: int,
    model_save_dir: str,
    save_prefix: str,
    safe_dataset: str,
    time_str: str,
    logger: logging.Logger,
    suppress_ratio: float = 0.2,
    **kwargs,
) -> TrainResult:
    """流程 5：HighCorClassifier，高相关 20%% 通道软正交抑制。"""
    return _run_classifier_pipeline(
        HighCorClassifier,
        "HighCorClassifier",
        model,
        train_caches,
        test_cache,
        train_size,
        ssc_output,
        class_number,
        classifier_iterations,
        classifier_lr,
        classifier_test_gap,
        model_save_dir,
        save_prefix,
        safe_dataset,
        time_str,
        logger,
        suppress_ratio=suppress_ratio,
        **kwargs,
    )


CLASSIFIER_TRAINERS: Dict[str, Callable[..., TrainResult]] = {
    "full": train_full_suppress_classifier,
    "none": train_no_suppress_classifier,
    "random20": train_random_suppress_classifier,
    "lowcor": train_low_cor_classifier,
    "highcor": train_high_cor_classifier,
}


def write_result_markdown(
    result_path: str,
    results: List[TrainResult],
    ssc_base_path: str,
    dataset_name: str,
    data_source: str,
    classifier_iteration: int,
    classifier_lr: float,
    append: bool = False,
    repeat_runs: int = 1,
) -> None:
    """将单次 run 的 best classifier accuracy 写入 markdown。"""
    os.makedirs(os.path.dirname(os.path.abspath(result_path)), exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    argv_summary = " ".join([sys.argv[0]] + sys.argv[1:])
    has_content = (
        append
        and os.path.isfile(result_path)
        and os.path.getsize(result_path) > 0
    )
    run_note = f", repeat_runs={repeat_runs}" if repeat_runs > 1 else ""
    lines = [
        "# Correlation-based Feature Suppression 实验",
        "",
        f"## {dataset_name} benchmark "
        f"(classifier_iteration={classifier_iteration}, classifier_lr={classifier_lr}{run_note}) — {timestamp}",
        "",
        f"_SSC-base: `{ssc_base_path}`_",
        f"_data_root: `{data_source}`_",
        "",
        f"_命令: `{argv_summary}`_",
        "",
        "| Classifier | Best Test Accuracy | Macro-F1 | Weighted-F1 | Balanced Acc | Checkpoint |",
        "|------------|-------------------|----------|-------------|--------------|------------|",
    ]
    for r in results:
        m = r.metrics
        run_tag = f" (run{r.run_idx + 1})" if repeat_runs > 1 else ""
        lines.append(
            f"| {r.classifier_name}{run_tag} | {m.get('accuracy', 0):.4f} | "
            f"{m.get('macro_f1', 0):.4f} | {m.get('weighted_f1', 0):.4f} | "
            f"{m.get('balanced_accuracy', 0):.4f} | `{r.classifier_path}` |"
        )
    lines.append("")
    body_lines = lines[2:] if has_content else lines
    mode = "a" if has_content else "w"
    with open(result_path, mode, encoding="utf-8") as f:
        if has_content:
            f.write("\n")
        f.write("\n".join(body_lines))


def write_aggregated_markdown(
    result_path: str,
    aggregated: List[AggregatedResult],
    ssc_base_path: str,
    dataset_name: str,
    data_source: str,
    classifier_iteration: int,
    classifier_lr: float,
    repeat_runs: int,
    append: bool = False,
) -> None:
    """将多次 run 的 mean±std 统计写入 markdown。"""
    os.makedirs(os.path.dirname(os.path.abspath(result_path)), exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    argv_summary = " ".join([sys.argv[0]] + sys.argv[1:])
    has_content = (
        append
        and os.path.isfile(result_path)
        and os.path.getsize(result_path) > 0
    )
    lines = [
        "# Correlation-based Feature Suppression 实验",
        "",
        f"## {dataset_name} benchmark — mean±std over {repeat_runs} runs "
        f"(classifier_iteration={classifier_iteration}, classifier_lr={classifier_lr}) — {timestamp}",
        "",
        f"_SSC-base: `{ssc_base_path}`_",
        f"_data_root: `{data_source}`_",
        "",
        f"_命令: `{argv_summary}`_",
        "",
        "| Classifier | Accuracy (mean±std) | Macro-F1 (mean±std) | "
        "Weighted-F1 (mean±std) | Balanced Acc (mean±std) |",
        "|------------|---------------------|---------------------|"
        "------------------------|-------------------------|",
    ]
    for agg in aggregated:
        m, s = agg.mean_metrics, agg.std_metrics
        lines.append(
            f"| {agg.classifier_name} | "
            f"{m['accuracy']:.4f}±{s['accuracy']:.4f} | "
            f"{m['macro_f1']:.4f}±{s['macro_f1']:.4f} | "
            f"{m['weighted_f1']:.4f}±{s['weighted_f1']:.4f} | "
            f"{m['balanced_accuracy']:.4f}±{s['balanced_accuracy']:.4f} |"
        )
    lines.append("")
    lines.append("<details><summary>各 run 明细 (Accuracy)</summary>")
    lines.append("")
    header = "| Classifier | " + " | ".join(f"Run {i + 1}" for i in range(repeat_runs)) + " |"
    sep = "|------------|" + "|".join(["----------"] * repeat_runs) + "|"
    lines.extend([header, sep])
    for agg in aggregated:
        run_accs = " | ".join(f"{r.metrics.get('accuracy', 0):.4f}" for r in agg.run_results)
        lines.append(f"| {agg.classifier_name} | {run_accs} |")
    lines.extend(["", "</details>", ""])
    body_lines = lines[2:] if has_content else lines
    mode = "a" if has_content else "w"
    with open(result_path, mode, encoding="utf-8") as f:
        if has_content:
            f.write("\n")
        f.write("\n".join(body_lines))


def resolve_ssc_base_path(dataset_name: str, ssc_base_path: Optional[str]) -> str:
    """单数据集可显式指定 base；多数据集 benchmark 从 BENCHMARK_SSC_BASE 读取。"""
    if ssc_base_path:
        return ssc_base_path
    if dataset_name not in BENCHMARK_SSC_BASE:
        raise ValueError(
            f"数据集 {dataset_name} 未配置 BENCHMARK_SSC_BASE，请通过 --ssc_base_path 指定"
        )
    return BENCHMARK_SSC_BASE[dataset_name]


def run_dataset_experiment(
    dataset_name: str,
    ssc_base_path: str,
    args: argparse.Namespace,
    classifier_iteration: int,
    classifier_lr: float,
    classifier_test_gap: int,
    batch_size: int,
    image_size: int,
    ssc_output: int,
    logger: logging.Logger,
    time_str: str,
) -> Tuple[List[AggregatedResult], List[List[TrainResult]]]:
    """对单个数据集重复训练 repeat_runs 次，返回聚合结果与各 run 明细。"""
    data_root = args.data_root.rstrip("/") + "/"
    data_source = os.path.join(data_root.rstrip("/"), dataset_name).rstrip("/") + "/"
    class_number = CLASS_NUM_DICT.get(dataset_name, 10)
    safe_name = dataset_name.replace("/", "_")
    feature_name = f"{safe_name}_resnet50"

    train_feat = load_dataFeatures(
        os.path.join(args.pre_feature_path, f"{feature_name}_train_features.pkl")
    )
    test_feat = load_dataFeatures(
        os.path.join(args.pre_feature_path, f"{feature_name}_test_features.pkl")
    )

    logger.info("SSC-base: %s", ssc_base_path)
    logger.info("dataset: %s (classes=%d)", data_source, class_number)
    model = load_frozen_ssc_base(ssc_base_path)

    logger.info("构建 SSC 特征缓存 (K=%d)...", args.cache_rounds)
    train_caches, test_cache, train_size, _ = build_ssc_caches(
        model,
        data_source,
        train_feat,
        test_feat,
        image_size,
        batch_size,
        cache_rounds=args.cache_rounds,
    )

    all_runs: List[List[TrainResult]] = []
    for run_idx in range(args.repeat_runs):
        set_run_seed(args.seed_base + run_idx)
        logger.info(
            "========== [%s] repeat run %d/%d (seed=%d) ==========",
            dataset_name,
            run_idx + 1,
            args.repeat_runs,
            args.seed_base + run_idx,
        )
        run_results: List[TrainResult] = []
        for mode in args.modes:
            trainer = CLASSIFIER_TRAINERS[mode]
            kwargs: Dict = {"run_idx": run_idx, "repeat_runs": args.repeat_runs}
            if mode in ("random20", "lowcor", "highcor"):
                kwargs["suppress_ratio"] = args.suppress_ratio
            mode_prefix = MODE_SAVE_PREFIX[mode]
            result = trainer(
                model,
                train_caches,
                test_cache,
                train_size,
                ssc_output,
                class_number,
                classifier_iteration,
                classifier_lr,
                classifier_test_gap,
                args.model_path.rstrip("/") + "/",
                mode_prefix,
                safe_name,
                time_str,
                logger,
                **kwargs,
            )
            run_results.append(result)
        all_runs.append(run_results)

    aggregated = aggregate_run_results(all_runs)
    return aggregated, all_runs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="加载 SSC-base，训练 IEEE 三种分类头（correlation-based feature suppression）"
    )
    p.add_argument(
        "--ssc_base_path",
        type=str,
        default=DEFAULT_SSC_BASE,
        help="已训练 SSC-base-best.pth 路径",
    )
    p.add_argument("--dataset_name", type=str, default="Painting91")
    p.add_argument("--data_root", type=str, default="/mnt/codes/data/style/")
    p.add_argument("--pre_feature_path", type=str, default="./pretrainFeatures")
    p.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_OUT,
        help="分类器 checkpoint 保存目录",
    )
    p.add_argument(
        "--result_md",
        type=str,
        default=DEFAULT_RESULT_MD,
        help="best accuracy 结果 markdown 路径",
    )
    p.add_argument(
        "--append_result",
        action="store_true",
        help="追加写入 result_md（默认覆盖）",
    )
    p.add_argument(
        "--modes",
        nargs="+",
        default=["full", "none", "random20", "lowcor", "highcor"],
        choices=list(CLASSIFIER_TRAINERS.keys()),
        help="要训练的分类器模式",
    )
    p.add_argument("--classifier_iteration", type=int, default=None)
    p.add_argument("--classifier_lr", type=float, default=None)
    p.add_argument("--classifier_test_gap", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--image_size", type=int, default=None)
    p.add_argument("--cache_rounds", type=int, default=12, help="train 增强缓存份数 K")
    p.add_argument(
        "--suppress_ratio",
        type=float,
        default=0.2,
        help="Random/LowCor/HighCor 通道抑制比例",
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        choices=list(BENCHMARK_SSC_BASE.keys()),
        help="多数据集 benchmark（默认使用 BENCHMARK_SSC_BASE 中对应 SSC-base）",
    )
    p.add_argument(
        "--repeat_runs",
        type=int,
        default=1,
        help="每个数据集重复训练次数（用于 mean±std 统计）",
    )
    p.add_argument(
        "--seed_base",
        type=int,
        default=42,
        help="repeat run 随机种子基数，第 i 次 run 使用 seed_base+i",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset_list = args.datasets if args.datasets else [args.dataset_name]

    train_args = argparse.Namespace(
        epochs=None,
        batch_size=args.batch_size,
        base_lr=None,
        image_size=args.image_size,
        classifier_iteration=args.classifier_iteration,
        classifier_lr=args.classifier_lr,
        classifier_training_gap=None,
        classifier_test_gap=args.classifier_test_gap,
    )
    (
        _,
        batch_size,
        _,
        _,
        image_size,
        classifier_iteration,
        classifier_lr,
        _,
        _,
        _,
        _,
        _,
        ssc_output,
        _,
        classifier_test_gap,
    ) = merge_params_with_args(parameter_load(), train_args)

    os.makedirs(args.model_path, exist_ok=True)
    os.makedirs("log", exist_ok=True)
    time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    multi_dataset = len(dataset_list) > 1 or args.datasets is not None

    for ds_idx, dataset_name in enumerate(dataset_list):
        # 单数据集且显式指定 --ssc_base_path 时优先使用；否则走 benchmark 映射
        if len(dataset_list) == 1 and not args.datasets:
            ssc_base_path = resolve_ssc_base_path(dataset_name, args.ssc_base_path)
        else:
            ssc_base_path = resolve_ssc_base_path(dataset_name, None)

        safe_name = dataset_name.replace("/", "_")
        log_path = os.path.join("log", f"ieee-cfs-{safe_name}-{time_str}.log")
        logger = _make_logger(log_path)

        data_root = args.data_root.rstrip("/") + "/"
        data_source = os.path.join(data_root.rstrip("/"), dataset_name).rstrip("/") + "/"

        aggregated, all_runs = run_dataset_experiment(
            dataset_name,
            ssc_base_path,
            args,
            classifier_iteration,
            classifier_lr,
            classifier_test_gap,
            batch_size,
            image_size,
            ssc_output,
            logger,
            time_str,
        )

        append_md = args.append_result or ds_idx > 0 or multi_dataset
        if args.repeat_runs > 1:
            write_aggregated_markdown(
                args.result_md,
                aggregated,
                ssc_base_path,
                dataset_name,
                data_source,
                classifier_iteration,
                classifier_lr,
                args.repeat_runs,
                append=append_md,
            )
        else:
            write_result_markdown(
                args.result_md,
                all_runs[0],
                ssc_base_path,
                dataset_name,
                data_source,
                classifier_iteration,
                classifier_lr,
                append=append_md,
                repeat_runs=1,
            )

        print(f"\n========== {dataset_name} Correlation-based Feature Suppression ==========")
        if args.repeat_runs > 1:
            for agg in aggregated:
                m, s = agg.mean_metrics, agg.std_metrics
                print(
                    f"{agg.classifier_name:28s} acc={m['accuracy']:.4f}±{s['accuracy']:.4f} "
                    f"macro_f1={m['macro_f1']:.4f}±{s['macro_f1']:.4f}"
                )
        else:
            for r in all_runs[0]:
                print(
                    f"{r.classifier_name:28s} acc={r.metrics.get('accuracy', 0):.4f} "
                    f"macro_f1={r.metrics.get('macro_f1', 0):.4f} "
                    f"-> {r.classifier_path}"
                )

    print(f"\n结果已写入: {args.result_md}")


if __name__ == "__main__":
    main()
