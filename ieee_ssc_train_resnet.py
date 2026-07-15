# SSC ResNet50 五数据集论文 benchmark（runs=3，四项指标）
# 训练逻辑与超参默认沿用 ssc_train_resnet_copy.py（parameter_load）
# 结果写入 ieee_access_paperdata/ours_multiple.md（格式对齐 BarlowTwins_multiple.md）
#
# 用法:
#   python ieee_ssc_train_resnet.py --benchmark_all
#   python ieee_ssc_train_resnet.py --dataset_name Painting91 --runs 3
#   ./ieee_access_codes/manage_ieee_ssc_train_bat.sh start   # 五库后台批量

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import numpy as np

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ssc_train_resnet_copy import (  # noqa: E402
    RunMetrics,
    SSCtrain,
    merge_params_with_args,
    parameter_load,
)

MODEL_NAME = "Ours (SSC-ResNet50)"
DEFAULT_DATA_BASE = "/mnt/codes/data/style"
DEFAULT_RESULT_MD = os.path.join(_ROOT, "ieee_access_paperdata", "ours_multiple.md")

BENCHMARK_DATASETS = [
    ("Painting91", 13, "Painting91"),
    ("Pandora", 12, "Pandora"),
    ("ArtBench", 10, "Artbench"),
    ("FashionStyle14", 14, "FashionStyle14"),
    ("Arch", 25, "Arch"),
]

DATASET_ORDER = ["Painting91", "Pandora", "ArtBench", "FashionStyle14", "Arch"]

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "macro_f1": "Macro-F1",
    "weighted_f1": "Weighted-F1",
    "balanced_accuracy": "Balanced Accuracy",
}


@dataclass
class DatasetResult:
    name: str
    num_classes: int
    data_root: str
    all_runs: List[RunMetrics]


def _format_mean_std(values: List[float]) -> str:
    valid = [v for v in values if not np.isnan(v)]
    if not valid:
        return "FAILED" if values else "-"
    arr = np.asarray(valid, dtype=np.float64)
    mean_v = float(np.mean(arr))
    std_v = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return f"{mean_v:.4f}±{std_v:.4f}"


def _run_cell_value(all_runs: List[RunMetrics], run_idx: int, metric_key: str) -> str:
    if run_idx >= len(all_runs):
        return "-"
    v = all_runs[run_idx][metric_key]
    if np.isnan(v):
        return "FAILED"
    return f"{v:.4f}"


def _format_metric_table_block(
    metric_title: str, results: List[DatasetResult], metric_key: str, runs: int
) -> List[str]:
    run_headers = [f"run{i}" for i in range(1, runs + 1)]
    lines = [
        f"### {metric_title}",
        "",
        "| Dataset | num_classes | "
        + " | ".join(run_headers)
        + " | mean±std | data_root |",
        "|" + "|".join(["---------"] * (4 + runs)) + "|",
    ]
    for result in results:
        values = [m[metric_key] for m in result.all_runs]
        run_cells = [_run_cell_value(result.all_runs, i, metric_key) for i in range(runs)]
        lines.append(
            f"| {result.name} | {result.num_classes} | "
            + " | ".join(run_cells)
            + f" | {_format_mean_std(values)} | `{result.data_root}` |"
        )
    lines.append("")
    return lines


def _format_summary_table(results: List[DatasetResult]) -> List[str]:
    metric_titles = list(METRIC_LABELS.values())
    lines = [
        "## 汇总总表",
        "",
        "| Dataset | num_classes | " + " | ".join(metric_titles) + " |",
        "|---------|-------------|" + "|".join(["---------"] * len(metric_titles)) + "|",
    ]
    for result in results:
        cells = [_format_mean_std([m[k] for m in result.all_runs]) for k in METRIC_LABELS]
        lines.append(f"| {result.name} | {result.num_classes} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def write_result_markdown(
    result_path: str,
    results: List[DatasetResult],
    data_base: str,
    ssc_epochs: int,
    classifier_iteration: int,
    runs: int,
    completed_runs: Optional[int] = None,
    append: bool = False,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(result_path)), exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    argv_summary = " ".join([sys.argv[0]] + sys.argv[1:])
    dataset_names = ", ".join(r.name for r in results)
    done = completed_runs if completed_runs is not None else max(
        (len(r.all_runs) for r in results), default=0
    )
    progress = f", completed={done}/{runs}" if done < runs else ""
    lines = [
        f"# {MODEL_NAME} 多数据集多次实验",
        "",
        f"## {MODEL_NAME} benchmark ({dataset_names}) "
        f"(ssc_epochs={ssc_epochs}, classifier_iteration={classifier_iteration}, "
        f"runs={runs}{progress}) — {timestamp}",
        "",
        f"_data_base=`{data_base}`_",
        "",
        f"_命令: `{argv_summary}`_",
        "",
    ]
    for metric_key, metric_title in METRIC_LABELS.items():
        lines.extend(_format_metric_table_block(metric_title, results, metric_key, runs))
    lines.extend(_format_summary_table(results))
    has_content = (
        append
        and os.path.isfile(result_path)
        and os.path.getsize(result_path) > 0
    )
    body_lines = lines[2:] if has_content else lines
    mode = "a" if has_content else "w"
    with open(result_path, mode, encoding="utf-8") as f:
        if has_content:
            f.write("\n")
        f.write("\n".join(body_lines))


def _build_ssc_args(ieee_args: argparse.Namespace) -> argparse.Namespace:
    """从 ieee 脚本参数构造 ssc_train_resnet_copy 所用 Namespace（默认 parameter_load）。"""
    return argparse.Namespace(
        dataset_name="Painting91",
        data_root=ieee_args.data_base.rstrip("/") + "/",
        pre_feature_path=ieee_args.pre_feature_path,
        model_path=ieee_args.model_path,
        training_mode="original",
        base_model_path="###",
        epochs=ieee_args.epochs,
        batch_size=ieee_args.batch_size,
        base_lr=ieee_args.base_lr,
        image_size=None,
        classifier_iteration=ieee_args.classifier_iteration,
        classifier_lr=None,
        classifier_training_gap=None,
        classifier_test_gap=None,
        iterations=1,
        dataset_repeat_runs=1,
    )


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


def _run_single_dataset(
    label: str,
    rel_dir: str,
    num_classes: int,
    args: argparse.Namespace,
    ssc_args: argparse.Namespace,
    ssc_epochs: int,
    classifier_iteration: int,
    write_incremental: bool = True,
    accumulated_results: Optional[List[DatasetResult]] = None,
    append_result: bool = False,
) -> DatasetResult:
    data_root = os.path.join(args.data_base.rstrip("/"), rel_dir)
    data_source = data_root.rstrip("/") + "/"
    safe_name = rel_dir.replace("/", "_")
    model_name = f"ssc-{safe_name}"
    feature_name = f"{safe_name}_resnet50"

    all_runs: List[RunMetrics] = []
    for run_idx in range(1, args.runs + 1):
        time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        log_path = os.path.join(
            "log", f"ieee-ssc-resnet50-{label}-run{run_idx}-{time_str}.log"
        )
        logger = _make_logger(log_path)
        current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

        print(f"[{label}] run {run_idx}/{args.runs} 开始训练…")
        try:
            ssc_args.dataset_name = rel_dir
            ssc_args.data_root = args.data_base.rstrip("/") + "/"
            ssc_args.dataset_repeat_runs = 1

            _, _, _, metrics_list = SSCtrain(
                logger,
                ssc_args.model_path.rstrip("/") + "/",
                current_time + f"-run{run_idx - 1}",
                model_name,
                data_source,
                num_classes,
                ssc_args.iterations,
                ssc_args.training_mode,
                ssc_args.base_model_path,
                ssc_args.pre_feature_path,
                feature_name,
                train_args=ssc_args,
                dataset_repeat_runs=1,
                collect_run_metrics=True,
            )
            metrics = metrics_list[0]
            all_runs.append(metrics)
            print(
                f"[{label}] run {run_idx}/{args.runs} "
                f"acc={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}, "
                f"weighted_f1={metrics['weighted_f1']:.4f}, "
                f"balanced_acc={metrics['balanced_accuracy']:.4f}"
            )
        except Exception as exc:
            print(f"[{label}] run {run_idx}/{args.runs} FAILED: {exc}")
            logger.exception("Run failed")
            all_runs.append(
                RunMetrics(
                    accuracy=float("nan"),
                    macro_f1=float("nan"),
                    weighted_f1=float("nan"),
                    balanced_accuracy=float("nan"),
                )
            )

        if write_incremental and args.result_md and not append_result:
            if accumulated_results is not None:
                md_results = list(accumulated_results) + [
                    DatasetResult(
                        name=label,
                        num_classes=num_classes,
                        data_root=data_root.rstrip("/"),
                        all_runs=all_runs,
                    )
                ]
            else:
                md_results = [
                    DatasetResult(
                        name=label,
                        num_classes=num_classes,
                        data_root=data_root.rstrip("/"),
                        all_runs=all_runs,
                    )
                ]
            write_result_markdown(
                args.result_md,
                md_results,
                args.data_base.rstrip("/") + "/",
                ssc_epochs,
                classifier_iteration,
                args.runs,
                completed_runs=len(all_runs),
                append=append_result,
            )
            print(f"结果已更新: {args.result_md} ({len(all_runs)}/{args.runs} runs)")

    return DatasetResult(
        name=label,
        num_classes=num_classes,
        data_root=data_root.rstrip("/"),
        all_runs=all_runs,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=f"{MODEL_NAME} 五库 benchmark")
    p.add_argument("--data_base", type=str, default=DEFAULT_DATA_BASE)
    p.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="单数据集模式：子目录名（如 Painting91）；与 --benchmark_all 二选一",
    )
    p.add_argument(
        "--benchmark_all",
        action="store_true",
        help="依次在五数据集上训练（Painting91, Pandora, ArtBench, FashionStyle14, Arch）",
    )
    p.add_argument(
        "--runs",
        type=int,
        default=3,
        help="每个数据集重复次数（默认 3）",
    )
    p.add_argument("--result_md", type=str, default=DEFAULT_RESULT_MD)
    p.add_argument(
        "--append_result",
        action="store_true",
        help="将结果追加写入 result_md（保留已有内容，如 Painting91 历史记录）",
    )
    p.add_argument(
        "--pre_feature_path",
        type=str,
        default="./pretrainFeatures",
        help="预提取 ResNet50 特征目录",
    )
    p.add_argument("--model_path", type=str, default="./model/")
    p.add_argument("--epochs", type=int, default=None, help="覆盖 ssc_train_resnet_copy parameter_load()")
    p.add_argument("--classifier_iteration", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--base_lr", type=float, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.runs < 1:
        raise SystemExit("错误: --runs 须 >= 1")

    os.makedirs(args.model_path, exist_ok=True)
    os.makedirs("log", exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.result_md)), exist_ok=True)

    ssc_args = _build_ssc_args(args)
    (
        epochs,
        _,
        _,
        _,
        _,
        classifier_iteration,
        *_,
    ) = merge_params_with_args(parameter_load(), ssc_args)

    results: List[DatasetResult] = []

    if args.benchmark_all:
        for label, n_cls, rel in BENCHMARK_DATASETS:
            result = _run_single_dataset(
                label,
                rel,
                n_cls,
                args,
                ssc_args,
                epochs,
                classifier_iteration,
                write_incremental=not args.append_result,
                accumulated_results=results,
                append_result=args.append_result,
            )
            results.append(result)
            md_results = [result] if args.append_result else results
            write_result_markdown(
                args.result_md,
                md_results,
                args.data_base.rstrip("/") + "/",
                epochs,
                classifier_iteration,
                args.runs,
                append=args.append_result,
            )
        print(f"全部完成，结果已{'追加' if args.append_result else '写入'}: {args.result_md}")
        return

    if not args.dataset_name:
        raise SystemExit("错误: 请指定 --dataset_name 或使用 --benchmark_all")

    matched = None
    for label, n_cls, rel in BENCHMARK_DATASETS:
        if args.dataset_name in (label, rel):
            matched = (label, n_cls, rel)
            break
    if matched is None:
        raise SystemExit(
            f"错误: 未知数据集 {args.dataset_name!r}，请使用五库之一: {DATASET_ORDER}"
        )

    label, n_cls, rel = matched
    result = _run_single_dataset(
        label, rel, n_cls, args, ssc_args, epochs, classifier_iteration,
        append_result=args.append_result,
    )
    write_result_markdown(
        args.result_md,
        [result],
        args.data_base.rstrip("/") + "/",
        epochs,
        classifier_iteration,
        args.runs,
        append=args.append_result,
    )
    print(f"结果已{'追加' if args.append_result else '写入'}: {args.result_md}")


if __name__ == "__main__":
    main()
