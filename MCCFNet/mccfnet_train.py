"""
MCCFNet：端到端训练与测试集评估；支持单数据集多次 run 或五数据集 benchmark。
"""
from __future__ import annotations

import argparse
import glob
import logging
import os
import re
import sys
import time
from argparse import Namespace
from dataclasses import dataclass
from datetime import datetime
from typing import TypedDict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import DenseNet169_Weights, densenet169

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_NAME = "MCCFNet"
DEFAULT_DATA_BASE = "/mnt/codes/data/style/"
DEFAULT_RESULT_MD = os.path.join(_ROOT, "ieee_access_paperdata", "MCCFNet_multiple.md")

# 与 run_simclr_train_bat.sh 五库一致
BENCHMARK_DATASETS: list[tuple[str, int, str]] = [
    ("Painting91", 13, "Painting91"),
    ("Pandora", 12, "Pandora"),
    ("Artbench", 10, "ArtBench"),
    ("FashionStyle14", 14, "FashionStyle14"),
    ("Arch", 25, "Arch"),
]

DATASET_ORDER = ["Painting91", "Pandora", "ArtBench", "FashionStyle14", "Arch"]
DATASET_REL = {
    "Painting91": "Painting91",
    "Pandora": "Pandora",
    "ArtBench": "Artbench",
    "FashionStyle14": "FashionStyle14",
    "Arch": "Arch",
}

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "macro_f1": "Macro-F1",
    "weighted_f1": "Weighted-F1",
    "balanced_accuracy": "Balanced Accuracy",
}


class RunMetrics(TypedDict):
    accuracy: float
    macro_f1: float
    weighted_f1: float
    balanced_accuracy: float


@dataclass
class DatasetResult:
    name: str
    num_classes: int
    data_root: str
    all_runs: list[RunMetrics]


class RegionalWeightedPooling(nn.Module):
    """区域加权池化 (RWP)：1x1 卷积生成空间权重，加权后全局平均池化。"""

    def __init__(self, in_channels: int):
        super().__init__()
        self.spatial_weight_conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.sigmoid(self.spatial_weight_conv(x))
        weighted_x = x * weights
        out = F.adaptive_avg_pool2d(weighted_x, (1, 1)).view(x.size(0), -1)
        return out


class MCCFNet(nn.Module):
    """多通道色彩融合网络：DenseNet169 骨干 + RWP + 线性分类头。"""

    def __init__(self, num_classes: int = 13, in_channels: int = 6):
        super().__init__()
        self.backbone = densenet169(weights=DenseNet169_Weights.IMAGENET1K_V1)
        original_conv = self.backbone.features.conv0
        if in_channels != 3:
            self.backbone.features.conv0 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=False,
            )
            with torch.no_grad():
                self.backbone.features.conv0.weight[:, :3] = original_conv.weight
                self.backbone.features.conv0.weight[:, 3:] = original_conv.weight.mean(
                    dim=1, keepdim=True
                )

        self.features = self.backbone.features
        num_features = self.backbone.classifier.in_features
        self.rwp = RegionalWeightedPooling(in_channels=num_features)
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        features = F.relu(features, inplace=True)
        pooled_features = self.rwp(features)
        return self.classifier(pooled_features)


class ToRgbHsv6Tensor:
    """PIL → (6, H, W)：RGB [0,1] 与 OpenCV HSV 各通道归一化到 [0,1] 后拼接。"""

    def __call__(self, pil_img) -> torch.Tensor:
        rgb = np.array(pil_img.convert("RGB"))
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        h = torch.from_numpy(hsv[:, :, 0]).float() / 179.0
        s = torch.from_numpy(hsv[:, :, 1]).float() / 255.0
        v = torch.from_numpy(hsv[:, :, 2]).float() / 255.0
        hsv_t = torch.stack([h, s, v], dim=0)
        return torch.cat([rgb_t, hsv_t], dim=0)


def build_transforms(image_size: int = 224) -> transforms.Compose:
    mean = [0.485, 0.456, 0.406, 0.5, 0.5, 0.5]
    std = [0.229, 0.224, 0.225, 0.25, 0.25, 0.25]
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            ToRgbHsv6Tensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


@torch.no_grad()
def evaluate_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> RunMetrics:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        pred = logits.argmax(dim=1)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(pred.cpu().tolist())
    if not y_true:
        return RunMetrics(
            accuracy=0.0, macro_f1=0.0, weighted_f1=0.0, balanced_accuracy=0.0
        )
    labels_arr = np.asarray(y_true, dtype=np.int64)
    preds_arr = np.asarray(y_pred, dtype=np.int64)
    labels_all = list(range(num_classes))
    return RunMetrics(
        accuracy=float(np.mean(labels_arr == preds_arr)),
        macro_f1=float(
            f1_score(labels_arr, preds_arr, average="macro", labels=labels_all, zero_division=0)
        ),
        weighted_f1=float(
            f1_score(labels_arr, preds_arr, average="weighted", labels=labels_all, zero_division=0)
        ),
        balanced_accuracy=float(balanced_accuracy_score(labels_arr, preds_arr)),
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        n += labels.size(0)
    return total_loss / max(n, 1)


def _format_mean_std(values: list[float]) -> str:
    valid = [v for v in values if not np.isnan(v)]
    if not valid:
        return "FAILED" if values else "-"
    arr = np.asarray(valid, dtype=np.float64)
    mean_v = float(np.mean(arr))
    std_v = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return f"{mean_v:.4f}±{std_v:.4f}"


def _run_cell_value(all_runs: list[RunMetrics], run_idx: int, metric_key: str) -> str:
    if run_idx >= len(all_runs):
        return "-"
    v = all_runs[run_idx][metric_key]
    if np.isnan(v):
        return "FAILED"
    return f"{v:.4f}"


def _format_metric_table_block(
    metric_title: str, results: list[DatasetResult], metric_key: str, runs: int
) -> list[str]:
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


def _format_summary_table(results: list[DatasetResult]) -> list[str]:
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
        cells = [_format_mean_std([m[k] for m in result.all_runs]) for k in METRIC_LABELS]
        lines.append(
            f"| {result.name} | {result.num_classes} | " + " | ".join(cells) + " |"
        )
    lines.append("")
    return lines


def write_result_markdown(
    result_path: str,
    results: list[DatasetResult],
    data_base: str,
    epochs: int,
    runs: int,
    completed_runs: int | None = None,
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
        f"(epochs={epochs}, runs={runs}{progress}) — {timestamp}",
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


def _save_run_markdown(
    result_md: str,
    dataset_name: str,
    num_classes: int,
    data_root: str,
    all_runs: list[RunMetrics],
    epochs: int,
    total_runs: int,
) -> None:
    data_base = os.path.dirname(data_root.rstrip("/")) + os.sep
    results = [
        DatasetResult(
            name=dataset_name,
            num_classes=num_classes,
            data_root=data_root.rstrip("/"),
            all_runs=all_runs,
        )
    ]
    write_result_markdown(
        result_md,
        results,
        data_base,
        epochs,
        total_runs,
        completed_runs=len(all_runs),
    )


def merge_batch_partials(
    partial_dir: str,
    merge_result_md: str,
    runs: int,
    data_base: str = DEFAULT_DATA_BASE,
) -> None:
    """合并 partial_dir 下各数据集 md → 五库总表。"""
    if not data_base.endswith("/"):
        data_base += "/"

    partials: dict[str, str] = {}
    for path in sorted(glob.glob(os.path.join(partial_dir, "*.md"))):
        name = os.path.splitext(os.path.basename(path))[0]
        with open(path, encoding="utf-8") as f:
            partials[name] = f.read()

    epochs = "?"
    if partials:
        m = re.search(r"\(epochs=(\d+),\s*runs=\d+", next(iter(partials.values())))
        if m:
            epochs = m.group(1)

    def extract_table_row(text: str, section: str) -> str:
        pat = rf"### {re.escape(section)}\s*\n\n(\|.+\|\n\|[-| ]+\|\n)(\|.+\|)"
        hit = re.search(pat, text)
        return hit.group(2).strip() if hit else ""

    def extract_summary_row(text: str) -> str:
        pat = r"## 汇总总表\s*\n\n(\|.+\|\n\|[-| ]+\|\n)(\|.+\|)"
        hit = re.search(pat, text)
        return hit.group(2).strip() if hit else ""

    metric_sections = list(METRIC_LABELS.values())
    run_headers = [f"run{i}" for i in range(1, runs + 1)]
    lines = [
        f"# {MODEL_NAME} 多数据集多次实验",
        "",
        f"## {MODEL_NAME} benchmark ({', '.join(DATASET_ORDER)}) "
        f"(epochs={epochs}, runs={runs}) — {datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        f"_data_base=`{data_base}`_",
        "",
        f"_命令: `./MCCFNet/run_mccfnet_train_bat.sh` → `mccfnet_train.py` × "
        f"{len(DATASET_ORDER)} 数据集_",
        "",
    ]

    for section in metric_sections:
        lines += [
            f"### {section}",
            "",
            "| Dataset | num_classes | "
            + " | ".join(run_headers)
            + " | mean±std | data_root |",
            "|" + "|".join(["---------"] * (4 + runs)) + "|",
        ]
        for ds in DATASET_ORDER:
            row = extract_table_row(partials.get(ds, ""), section)
            if row:
                lines.append(row)
            else:
                failed = " | ".join(["FAILED"] * runs)
                lines.append(
                    f"| {ds} | ? | {failed} | FAILED | `{data_base}{DATASET_REL.get(ds, ds)}` |"
                )
        lines.append("")

    lines += [
        "## 汇总总表",
        "",
        "| Dataset | num_classes | Accuracy | Macro-F1 | Weighted-F1 | Balanced Accuracy |",
        "|---------|-------------|---------|---------|---------|---------|",
    ]
    for ds in DATASET_ORDER:
        row = extract_summary_row(partials.get(ds, ""))
        if row:
            lines.append(row)
        else:
            lines.append(f"| {ds} | ? | FAILED | FAILED | FAILED | FAILED |")
    lines.append("")

    os.makedirs(os.path.dirname(os.path.abspath(merge_result_md)), exist_ok=True)
    with open(merge_result_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def setup_torch_hub(root: str) -> None:
    hub_dir = os.path.join(root, "pretrainModels", "hub")
    os.makedirs(hub_dir, exist_ok=True)
    torch.hub.set_dir(hub_dir)
    os.environ.setdefault("TORCH_HOME", os.path.join(root, "pretrainModels"))


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


def train_single_dataset(args: Namespace, logger: logging.Logger) -> RunMetrics:
    """训练单个数据集，返回 test 集上 accuracy 最优 epoch 的四项指标。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = build_transforms(args.image_size)
    train_dir = os.path.join(args.data_root, "train")
    test_dir = os.path.join(args.data_root, "test")
    train_set = datasets.ImageFolder(train_dir, transform=transform)
    test_set = datasets.ImageFolder(test_dir, transform=transform)

    num_classes = args.num_classes if args.num_classes > 0 else len(train_set.classes)
    if len(train_set.classes) != num_classes:
        raise ValueError(
            f"train 目录推断类别数={len(train_set.classes)} 与 num_classes={num_classes} 不一致"
        )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = MCCFNet(num_classes=num_classes, in_channels=6).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    dataset_name = os.path.basename(os.path.normpath(args.data_root))
    time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    base_name = f"mccfnet-densenet169-{dataset_name}-{time_str}"

    logger.info(
        "data_root=%s, num_classes=%d, train=%d, test=%d",
        args.data_root,
        num_classes,
        len(train_set),
        len(test_set),
    )
    logger.info(
        "batch_size=%d, epochs=%d, lr=%g, weight_decay=%g, device=%s",
        args.batch_size,
        args.epochs,
        args.lr,
        args.weight_decay,
        device,
    )

    best_metrics = RunMetrics(
        accuracy=0.0, macro_f1=0.0, weighted_f1=0.0, balanced_accuracy=0.0
    )
    best_path = os.path.join(args.save_dir, f"{base_name}-best.pth")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        metrics = evaluate_metrics(model, test_loader, device, num_classes)
        logger.info(
            "Epoch [%d/%d] train_loss=%.4f acc=%.4f macro_f1=%.4f weighted_f1=%.4f balanced_acc=%.4f",
            epoch,
            args.epochs,
            train_loss,
            metrics["accuracy"],
            metrics["macro_f1"],
            metrics["weighted_f1"],
            metrics["balanced_accuracy"],
        )
        if metrics["accuracy"] > best_metrics["accuracy"]:
            best_metrics = metrics
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "metrics": dict(metrics),
                    "num_classes": num_classes,
                    "class_to_idx": train_set.class_to_idx,
                    "args": vars(args),
                },
                best_path,
            )
            logger.info("Saved best checkpoint: %s (acc=%.4f)", best_path, metrics["accuracy"])

    logger.info(
        "Done. acc=%.4f macro_f1=%.4f weighted_f1=%.4f balanced_acc=%.4f",
        best_metrics["accuracy"],
        best_metrics["macro_f1"],
        best_metrics["weighted_f1"],
        best_metrics["balanced_accuracy"],
    )
    print(
        f"Best test: acc={best_metrics['accuracy']:.4f}, macro_f1={best_metrics['macro_f1']:.4f}, "
        f"weighted_f1={best_metrics['weighted_f1']:.4f}, "
        f"balanced_acc={best_metrics['balanced_accuracy']:.4f}  (checkpoint: {best_path})"
    )
    return best_metrics


def parse_args() -> Namespace:
    p = argparse.ArgumentParser(description="MCCFNet 分类训练（单数据集或五数据集 benchmark）")
    p.add_argument(
        "--data_root",
        type=str,
        default="/mnt/codes/data/style/Painting91",
        help="数据集根目录（含 train/test）；benchmark 模式下由 data_base+子目录覆盖",
    )
    p.add_argument("--num_classes", type=int, default=13, help="<=0 时从 train 推断")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=3, help="端到端训练轮数")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--save_dir", type=str, default="model")
    p.add_argument(
        "--benchmark_all",
        action="store_true",
        help="依次在五数据集上训练（data_base 下子目录同 simclr/barlowtwins）",
    )
    p.add_argument("--data_base", type=str, default=DEFAULT_DATA_BASE)
    p.add_argument("--result_md", type=str, default=DEFAULT_RESULT_MD)
    p.add_argument(
        "--merge_result_md",
        type=str,
        default=None,
        help="批量模式：每轮 run 后合并 partial 写入此总表",
    )
    p.add_argument(
        "--partial_dir",
        type=str,
        default=None,
        help="各数据集 partial md 所在目录（配合 --merge_result_md）",
    )
    p.add_argument(
        "--dataset_label",
        type=str,
        default=None,
        help="批量脚本中的数据集显示名（如 ArtBench）",
    )
    p.add_argument(
        "--run",
        "--runs",
        type=int,
        default=3,
        dest="num_runs",
        metavar="N",
        help="每个数据集重复次数（默认 3），记录 mean±std",
    )
    p.add_argument(
        "--benchmark_runs",
        type=int,
        default=None,
        help="--benchmark_all 时重复次数；未指定时同 --runs",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = _ROOT
    os.chdir(root)
    setup_torch_hub(root)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs("log", exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.result_md)), exist_ok=True)

    if args.num_runs < 1:
        raise SystemExit("错误: --runs 须 >= 1")

    benchmark_runs = args.benchmark_runs if args.benchmark_runs is not None else args.num_runs

    if args.benchmark_all:
        if benchmark_runs < 1:
            raise ValueError("--benchmark_runs 须 >= 1")
        results: list[DatasetResult] = []
        data_base = os.path.normpath(args.data_base)
        for rel, n_cls, label in BENCHMARK_DATASETS:
            args.data_root = os.path.join(data_base, rel.replace("/", os.sep))
            args.num_classes = n_cls
            all_runs: list[RunMetrics] = []
            for run_idx in range(1, benchmark_runs + 1):
                time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
                log_path = os.path.join(
                    "log",
                    f"mccfnet-benchmark-{label}-run{run_idx}-{time_str}.log",
                )
                logger = _make_logger(log_path)
                try:
                    metrics = train_single_dataset(args, logger)
                    all_runs.append(metrics)
                    print(
                        f"[{label}] run {run_idx}/{benchmark_runs} "
                        f"acc={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}"
                    )
                except Exception as e:
                    logger.exception("Dataset run failed: %s run %d", label, run_idx)
                    all_runs.append(
                        RunMetrics(
                            accuracy=float("nan"),
                            macro_f1=float("nan"),
                            weighted_f1=float("nan"),
                            balanced_accuracy=float("nan"),
                        )
                    )
                    print(f"[{label}] run {run_idx}/{benchmark_runs} FAILED: {e}")
            results.append(
                DatasetResult(
                    name=label,
                    num_classes=n_cls,
                    data_root=os.path.abspath(args.data_root),
                    all_runs=all_runs,
                )
            )
        write_result_markdown(
            args.result_md,
            results,
            data_base + os.sep,
            args.epochs,
            benchmark_runs,
        )
        print(f"结果已写入: {args.result_md}")
        return

    if not args.data_root:
        raise SystemExit("错误: 必须指定 --data_root（或使用 --benchmark_all）")

    data_root = os.path.abspath(args.data_root.rstrip("/"))
    dataset_name = args.dataset_label or os.path.basename(os.path.normpath(data_root))
    all_runs: list[RunMetrics] = []
    for r in range(1, args.num_runs + 1):
        time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        log_path = os.path.join(
            "log", f"mccfnet-densenet169-{dataset_name}-run{r}-{time_str}.log"
        )
        logger = _make_logger(log_path)
        print(f"[{dataset_name} run{r}/{args.num_runs}] 开始训练…")
        try:
            metrics = train_single_dataset(args, logger)
            all_runs.append(metrics)
            print(
                f"[{dataset_name} run{r}/{args.num_runs}] "
                f"acc={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}, "
                f"weighted_f1={metrics['weighted_f1']:.4f}, "
                f"balanced_acc={metrics['balanced_accuracy']:.4f}"
            )
        except Exception as e:
            logger.exception("Run %d failed", r)
            all_runs.append(
                RunMetrics(
                    accuracy=float("nan"),
                    macro_f1=float("nan"),
                    weighted_f1=float("nan"),
                    balanced_accuracy=float("nan"),
                )
            )
            print(f"[{dataset_name} run{r}/{args.num_runs}] FAILED: {e}")

        _save_run_markdown(
            args.result_md,
            dataset_name,
            args.num_classes,
            data_root,
            all_runs,
            args.epochs,
            args.num_runs,
        )
        print(f"结果已更新: {args.result_md} ({len(all_runs)}/{args.num_runs} runs)")

        if args.merge_result_md and args.partial_dir:
            merge_batch_partials(
                args.partial_dir,
                args.merge_result_md,
                args.num_runs,
                args.data_base,
            )
            print(f"总表已更新: {args.merge_result_md}")

    print(
        f"[{dataset_name}] Accuracy "
        f"{_format_mean_std([m['accuracy'] for m in all_runs])}"
    )


if __name__ == "__main__":
    main()
