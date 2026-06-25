#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 ImageNet 预训练 ViT-L/16 的风格分类测试脚本。

用法:
    python vit_l_16_test.py --benchmark_all
    python vit_l_16_test.py --data_root /mnt/codes/data/style/Painting91 --num_classes 13 --epochs 15 --runs 5
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import TypedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


# 模型显示名（Markdown / 日志）
MODEL_NAME = "ViT-L/16"

# ImageNet 归一化参数
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INPUT_SIZE = 224
DEFAULT_DATA_BASE = "/mnt/codes/data/style"

# 五项 benchmark 数据集：(显示名, 类别数, 相对 data_base 的子目录)
BENCHMARK_DATASETS: list[tuple[str, int, str]] = [
    ("Painting91", 13, "Painting91"),
    ("Pandora", 12, "Pandora"),
    ("ArtBench", 10, "artbench-10-imagefolder-split"),
    ("FashionStyle14", 14, "FashionStyle14"),
    ("Arch", 25, "Arch"),
]

# 四项评测指标（Markdown 表头显示名）
METRIC_LABELS: dict[str, str] = {
    "accuracy": "Accuracy",
    "macro_f1": "Macro-F1",
    "weighted_f1": "Weighted-F1",
    "balanced_accuracy": "Balanced Accuracy",
}


class RunMetrics(TypedDict):
    """单次 run 在测试集上的四项指标。"""
    accuracy: float
    macro_f1: float
    weighted_f1: float
    balanced_accuracy: float


@dataclass
class DatasetResult:
    """单个数据集多次实验的汇总结果。"""
    name: str
    num_classes: int
    data_root: str
    all_runs: list[RunMetrics]


def get_project_root() -> str:
    """返回项目根目录（本脚本所在目录）。"""
    return os.path.dirname(os.path.abspath(__file__))


def get_pretrain_root() -> str:
    """预训练权重根目录：./pretrainModels/"""
    return os.path.join(get_project_root(), "pretrainModels")


def find_local_vit_l_16_checkpoint(checkpoint_dir: str) -> str | None:
    """在 pretrainModels/hub/checkpoints 下查找 ViT-L/16 权重文件。"""
    if not os.path.isdir(checkpoint_dir):
        return None
    matches = sorted(glob.glob(os.path.join(checkpoint_dir, "vit_l_16*.pth")))
    return matches[0] if matches else None


def _load_state_dict(checkpoint_path: str) -> dict:
    """兼容不同 PyTorch 版本的权重加载。"""
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")


def load_pretrained_vit_l_16(device: torch.device, verbose: bool = True) -> models.VisionTransformer:
    """
    加载 ImageNet 预训练 ViT-L/16：
    1. 优先从 ./pretrainModels/hub/checkpoints 读取本地权重；
    2. 若不存在，则下载至 ./pretrainModels/（由 TORCH_HOME 控制）。
    """
    pretrain_root = get_pretrain_root()
    checkpoint_dir = os.path.join(pretrain_root, "hub", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.environ["TORCH_HOME"] = pretrain_root

    local_ckpt = find_local_vit_l_16_checkpoint(checkpoint_dir)
    if local_ckpt is not None:
        if verbose:
            print(f"[{MODEL_NAME}] 使用本地预训练权重: {local_ckpt}")
        model = models.vit_l_16(weights=None)
        model.load_state_dict(_load_state_dict(local_ckpt))
    else:
        if verbose:
            print(f"[{MODEL_NAME}] 本地未找到权重，下载至: {pretrain_root}")
        model = models.vit_l_16(weights=models.ViT_L_16_Weights.IMAGENET1K_V1)

    return model.to(device)


def build_vit_l_16_classifier(num_classes: int, device: torch.device) -> models.VisionTransformer:
    """
    构建 ViT-L/16 分类器：
    - Transformer 编码器冻结；
    - 替换 heads.head 为 num_classes 输出；
    - 仅微调分类头。
    """
    model = load_pretrained_vit_l_16(device, verbose=False)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes).to(device)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.heads.parameters():
        param.requires_grad = True

    return model


def _set_backbone_eval(model: models.VisionTransformer) -> None:
    """冻结编码器时保持 LayerNorm 处于 eval，仅 heads 参与训练。"""
    model.eval()
    model.heads.train()


def build_dataloaders(
    data_root: str,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, int]:
    """构建 train/test DataLoader，返回 (train_loader, test_loader, num_classes)。"""
    transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_dir = os.path.join(data_root, "train")
    test_dir = os.path.join(data_root, "test")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"训练集目录不存在: {train_dir}")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"测试集目录不存在: {test_dir}")

    train_set = datasets.ImageFolder(train_dir, transform=transform)
    test_set = datasets.ImageFolder(test_dir, transform=transform)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader, len(train_set.classes)


@torch.no_grad()
def evaluate_metrics(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> RunMetrics:
    """在指定数据集上计算 Accuracy / Macro-F1 / Weighted-F1 / Balanced Accuracy。"""
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []

    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        pred = logits.argmax(dim=1)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(pred.cpu().tolist())

    if not y_true:
        return RunMetrics(
            accuracy=0.0,
            macro_f1=0.0,
            weighted_f1=0.0,
            balanced_accuracy=0.0,
        )

    labels_arr = np.asarray(y_true, dtype=np.int64)
    preds_arr = np.asarray(y_pred, dtype=np.int64)
    labels_all = list(range(num_classes))

    return RunMetrics(
        accuracy=float(np.mean(labels_arr == preds_arr)),
        macro_f1=float(f1_score(labels_arr, preds_arr, average="macro", labels=labels_all, zero_division=0)),
        weighted_f1=float(f1_score(labels_arr, preds_arr, average="weighted", labels=labels_all, zero_division=0)),
        balanced_accuracy=float(balanced_accuracy_score(labels_arr, preds_arr)),
    )


def train_one_epoch(
    model: models.VisionTransformer,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    """训练一个 epoch，返回平均 loss。"""
    _set_backbone_eval(model)

    running_loss = 0.0
    total = 0
    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        total += batch_size

    return running_loss / total if total > 0 else 0.0


def train_single_run(
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_classes: int,
    device: torch.device,
    epochs: int,
    lr: float,
) -> RunMetrics:
    """单次完整训练，返回测试集四项指标（多 epoch 时取 Accuracy 最高 epoch）。"""
    model = build_vit_l_16_classifier(num_classes, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.heads.parameters(), lr=lr)

    best_metrics = RunMetrics(
        accuracy=0.0,
        macro_f1=0.0,
        weighted_f1=0.0,
        balanced_accuracy=0.0,
    )

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        metrics = evaluate_metrics(model, test_loader, device, num_classes)
        if metrics["accuracy"] >= best_metrics["accuracy"]:
            best_metrics = metrics
        print(
            f"  [Epoch {epoch:03d}/{epochs}] loss={train_loss:.4f}, "
            f"acc={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}, "
            f"weighted_f1={metrics['weighted_f1']:.4f}, "
            f"balanced_acc={metrics['balanced_accuracy']:.4f}"
        )

    return best_metrics


def compute_mean_std(values: list[float]) -> tuple[float, float]:
    """计算 mean±std（样本标准差）。"""
    arr = np.asarray(values, dtype=np.float64)
    mean_v = float(np.mean(arr))
    std_v = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return mean_v, std_v


def run_dataset_benchmark(
    dataset_name: str,
    num_classes: int,
    data_root: str,
    device: torch.device,
    epochs: int,
    runs: int,
    batch_size: int,
    lr: float,
    num_workers: int,
) -> DatasetResult:
    """在单个数据集上重复 runs 次实验。"""
    print("=" * 60)
    print(f"数据集: {dataset_name} | 路径: {data_root} | 类别数: {num_classes}")

    train_loader, test_loader, detected_classes = build_dataloaders(
        data_root, batch_size, num_workers
    )
    if num_classes != detected_classes:
        print(f"[警告] 指定类别数 {num_classes} 与检测到 {detected_classes} 不一致，使用 {detected_classes}")
        num_classes = detected_classes

    print(f"训练样本: {len(train_loader.dataset)}, 测试样本: {len(test_loader.dataset)}")
    load_pretrained_vit_l_16(device, verbose=True)

    all_runs: list[RunMetrics] = []
    for run_idx in range(1, runs + 1):
        print("-" * 60)
        print(f"[{dataset_name}] Run {run_idx}/{runs} 开始")
        metrics = train_single_run(
            train_loader,
            test_loader,
            num_classes,
            device,
            epochs,
            lr,
        )
        all_runs.append(metrics)
        print(
            f"[{dataset_name}] Run {run_idx}/{runs} | "
            f"acc={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}, "
            f"weighted_f1={metrics['weighted_f1']:.4f}, "
            f"balanced_acc={metrics['balanced_accuracy']:.4f}"
        )

    return DatasetResult(
        name=dataset_name,
        num_classes=num_classes,
        data_root=os.path.abspath(data_root),
        all_runs=all_runs,
    )


def _format_mean_std(values: list[float]) -> str:
    mean_v, std_v = compute_mean_std(values)
    return f"{mean_v:.4f}±{std_v:.4f}" if not np.isnan(mean_v) else "FAILED"


def _format_metric_table_block(
    metric_title: str,
    results: list[DatasetResult],
    metric_key: str,
    runs: int,
) -> list[str]:
    """生成某一指标下所有数据集的详细表格。"""
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
        values = [run_metrics[metric_key] for run_metrics in result.all_runs]
        run_cells = [f"{v:.4f}" if not np.isnan(v) else "FAILED" for v in values]
        lines.append(
            f"| {result.name} | {result.num_classes} | "
            + " | ".join(run_cells)
            + f" | {_format_mean_std(values)} | `{result.data_root}` |"
        )

    lines.append("")
    return lines


def _format_summary_table(results: list[DatasetResult]) -> list[str]:
    """生成五数据集汇总总表（每指标一列 mean±std）。"""
    metric_titles = list(METRIC_LABELS.values())
    lines = [
        "## 汇总总表",
        "",
        "| Dataset | num_classes | "
        + " | ".join(metric_titles)
        + " |",
        "|---------|-------------|"
        + "|".join(["---------"] * len(metric_titles))
        + "|",
    ]

    for result in results:
        cells = []
        for metric_key in METRIC_LABELS:
            values = [run_metrics[metric_key] for run_metrics in result.all_runs]
            cells.append(_format_mean_std(values))
        lines.append(
            f"| {result.name} | {result.num_classes} | "
            + " | ".join(cells)
            + " |"
        )

    lines.append("")
    return lines


def write_result_markdown(
    result_path: str,
    results: list[DatasetResult],
    data_base: str,
    epochs: int,
    runs: int,
) -> None:
    """写入各数据集分项表 + 汇总总表。"""
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    argv_summary = " ".join([sys.argv[0]] + sys.argv[1:])
    dataset_names = ", ".join(r.name for r in results)

    lines = [
        f"# {MODEL_NAME} 多数据集多次实验",
        "",
        f"## {MODEL_NAME} benchmark ({dataset_names}) (epochs={epochs}, runs={runs}) — {timestamp}",
        "",
        f"_data_base=`{data_base}`_",
        "",
        f"_命令: `{argv_summary}`_",
        "",
    ]

    for metric_key, metric_title in METRIC_LABELS.items():
        lines.extend(_format_metric_table_block(metric_title, results, metric_key, runs))

    lines.extend(_format_summary_table(results))

    with open(result_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def print_dataset_summary(result: DatasetResult) -> None:
    """打印单个数据集的四项指标 mean±std。"""
    print(f"\n[{result.name}] 本次实验结果 (mean±std):")
    print("| 指标 | mean±std |")
    print("|------|----------|")
    for metric_key, metric_title in METRIC_LABELS.items():
        values = [m[metric_key] for m in result.all_runs]
        print(f"| {metric_title} | {_format_mean_std(values)} |")


def print_overall_summary(results: list[DatasetResult]) -> None:
    """打印汇总总表到终端。"""
    print("\n" + "=" * 60)
    print("汇总总表 (mean±std)")
    print("| Dataset | num_classes | Accuracy | Macro-F1 | Weighted-F1 | Balanced Accuracy |")
    print("|---------|-------------|----------|----------|-------------|-------------------|")
    for result in results:
        cells = []
        for metric_key in METRIC_LABELS:
            values = [m[metric_key] for m in result.all_runs]
            cells.append(_format_mean_std(values))
        print(
            f"| {result.name} | {result.num_classes} | "
            + " | ".join(cells)
            + " |"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"{MODEL_NAME} 预训练风格分类测试")
    parser.add_argument(
        "--benchmark_all",
        action="store_true",
        help="依次评测 Painting91 / Pandora / ArtBench / FashionStyle14 / Arch",
    )
    parser.add_argument(
        "--data_base",
        type=str,
        default=DEFAULT_DATA_BASE,
        help="多数据集 benchmark 的数据根目录",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="单数据集模式：数据集根目录（需含 train/test 子目录）",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=0,
        help="单数据集模式下的类别数；<=0 时从数据集自动推断",
    )
    parser.add_argument("--epochs", type=int, default=15, help="微调 epoch 数")
    parser.add_argument("--runs", type=int, default=5, help="重复实验次数，报告 mean±std")
    parser.add_argument("--batch_size", type=int, default=32, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="分类头学习率")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader 线程数")
    parser.add_argument(
        "--result_md",
        type=str,
        default=os.path.join(get_project_root(), "ieee_access_paperdata", "vit_l_16_multiple.md"),
        help="多次实验结果 Markdown 输出路径",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.runs < 1:
        raise SystemExit("错误: --runs 须 >= 1")

    benchmark_all = args.benchmark_all or args.data_root is None
    if benchmark_all and args.data_root is not None:
        raise SystemExit("错误: --benchmark_all 与 --data_root 不能同时使用")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"模型: {MODEL_NAME}")
    print(f"设备: {device}")
    print(f"微调轮数: {args.epochs}, 重复次数: {args.runs}")

    t0 = time.time()
    results: list[DatasetResult] = []

    if benchmark_all:
        data_base = os.path.normpath(args.data_base)
        for name, num_classes, rel_path in BENCHMARK_DATASETS:
            data_root = os.path.join(data_base, rel_path.replace("/", os.sep))
            try:
                result = run_dataset_benchmark(
                    name,
                    num_classes,
                    data_root,
                    device,
                    args.epochs,
                    args.runs,
                    args.batch_size,
                    args.lr,
                    args.num_workers,
                )
                results.append(result)
                print_dataset_summary(result)
            except Exception as exc:
                print(f"[{name}] 失败: {exc}")
    else:
        data_root = args.data_root
        dataset_name = os.path.basename(os.path.normpath(data_root))
        num_classes = args.num_classes
        if num_classes <= 0:
            _, _, num_classes = build_dataloaders(data_root, args.batch_size, args.num_workers)
        result = run_dataset_benchmark(
            dataset_name,
            num_classes,
            data_root,
            device,
            args.epochs,
            args.runs,
            args.batch_size,
            args.lr,
            args.num_workers,
        )
        results.append(result)
        print_dataset_summary(result)

    if not results:
        raise SystemExit("错误: 没有成功完成任何数据集实验")

    elapsed = time.time() - t0
    data_base_for_md = os.path.normpath(args.data_base) if benchmark_all else os.path.dirname(results[0].data_root)

    write_result_markdown(
        args.result_md,
        results,
        data_base_for_md + os.sep,
        args.epochs,
        args.runs,
    )

    if len(results) > 1:
        print_overall_summary(results)

    print(f"\n结果已写入: {args.result_md}")
    print(f"总耗时: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
