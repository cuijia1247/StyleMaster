"""
基于 barlowtwins/barlow.py 中的 Barlow Twins 与 barlow_loss_fun：
1) 双视图 Barlow Twins 自监督预训练；2) 冻结 encoder，线性分类头在 test 上报告四项指标。

五数据集 benchmark（Painting91, Pandora, ArtBench, FashionStyle14, Arch），
结果写入 ieee_access_paperdata/BarlowTwins_multiple.md（格式对齐 vgg16_multiple.md）。

用法::
    python selfsupervised/barlowtwins_train.py --data_root /path/to/Painting91 --num_classes 13
    ./selfsupervised/run_barlowtwins_train_bat.sh
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, TypedDict

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from barlowtwins.barlow import BarlowTwins, barlow_loss_fun  # noqa: E402

MODEL_NAME = "Barlow Twins"
DEFAULT_DATA_BASE = "/mnt/codes/data/style"

BENCHMARK_DATASETS: list[tuple[str, int, str]] = [
    ("Painting91", 13, "Painting91"),
    ("Pandora", 12, "Pandora"),
    ("ArtBench", 10, "Artbench"),
    ("FashionStyle14", 14, "FashionStyle14"),
    ("Arch", 25, "Arch"),
]

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "macro_f1": "Macro-F1",
    "weighted_f1": "Weighted-F1",
    "balanced_accuracy": "Balanced Accuracy",
}

DEFAULT_RESULT_MD = os.path.join(
    _ROOT, "ieee_access_paperdata", "BarlowTwins_multiple.md"
)

DATASET_ORDER = ["Painting91", "Pandora", "ArtBench", "FashionStyle14", "Arch"]
DATASET_REL = {
    "Painting91": "Painting91",
    "Pandora": "Pandora",
    "ArtBench": "Artbench",
    "FashionStyle14": "FashionStyle14",
    "Arch": "Arch",
}


class RunMetrics(TypedDict):
    accuracy: float
    macro_f1: float
    weighted_f1: float
    balanced_accuracy: float


class RunEpochTimes(TypedDict):
    pretrain_s: list[float]
    classifier_s: list[float]


@dataclass
class DatasetResult:
    name: str
    num_classes: int
    data_root: str
    all_runs: list[RunMetrics]


def _build_barlow_twins_imagenet_encoder(
    device: torch.device, proj_channels: int
) -> BarlowTwins:
    """
    使用 barlow.py 的 BarlowTwins 结构，并将 encoder 换为 ImageNet 预训练 ResNet50（fc=Identity）。
    """
    in_features = 2048
    model = BarlowTwins(in_features, proj_channels).to(device)
    try:
        w = models.ResNet50_Weights.IMAGENET1K_V1
        r = models.resnet50(weights=w)
    except Exception:
        r = models.resnet50(pretrained=True)
    r.fc = nn.Identity()
    model.encoder.load_state_dict(r.state_dict())
    return model


class TwoViewImageFolder(Dataset):
    """同一张图两次随机增强，返回 (view1, view2, label)。"""

    def __init__(self, root: str, transform: transforms.Compose):
        self.base = datasets.ImageFolder(root, transform=None)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int):
        path, label = self.base.samples[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), self.transform(img), label


def build_base_name(data_root: str) -> str:
    dataset_name = os.path.basename(os.path.normpath(data_root))
    time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    return f"barlowtwins-resnet50-{dataset_name}-{time_str}"


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


@torch.no_grad()
def evaluate_probe_metrics(
    bt: nn.Module,
    classifier: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> RunMetrics:
    bt.eval()
    classifier.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        h = bt.encoder(images)
        logits = classifier(h)
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
    pretrain_epochs: int,
    classifier_epochs: int,
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
        f"(pretrain_epochs={pretrain_epochs}, classifier_epochs={classifier_epochs}, "
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
    with open(result_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _save_run_markdown(
    result_md: str,
    dataset_name: str,
    num_classes: int,
    data_root: str,
    all_runs: list[RunMetrics],
    pretrain_epochs: int,
    classifier_epochs: int,
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
        pretrain_epochs,
        classifier_epochs,
        total_runs,
        completed_runs=len(all_runs),
    )


def merge_batch_partials(
    partial_dir: str,
    merge_result_md: str,
    runs: int,
    data_base: str = DEFAULT_DATA_BASE,
) -> None:
    """合并 partial_dir 下各数据集 md → 五库总表（格式对齐 vgg16_multiple.md）。"""
    if not data_base.endswith("/"):
        data_base += "/"

    partials: dict[str, str] = {}
    for path in sorted(glob.glob(os.path.join(partial_dir, "*.md"))):
        name = os.path.splitext(os.path.basename(path))[0]
        with open(path, encoding="utf-8") as f:
            partials[name] = f.read()

    pretrain_epochs = classifier_epochs = "?"
    if partials:
        m = re.search(
            r"pretrain_epochs=(\d+),\s*classifier_epochs=(\d+)",
            next(iter(partials.values())),
        )
        if m:
            pretrain_epochs, classifier_epochs = m.group(1), m.group(2)

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
        f"(pretrain_epochs={pretrain_epochs}, classifier_epochs={classifier_epochs}, "
        f"runs={runs}) — {datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        f"_data_base=`{data_base}`_",
        "",
        "_命令: `./selfsupervised/run_barlowtwins_train_bat.sh` → `barlowtwins_train.py` × "
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


def write_run_epoch_times(
    epoch_time_dir: str | None,
    dataset_label: str,
    run_index: int,
    epoch_times: RunEpochTimes,
    logger: logging.Logger,
) -> None:
    """每轮 run 结束后记录各 epoch 耗时（秒）。"""
    pretrain_times = epoch_times["pretrain_s"]
    classifier_times = epoch_times["classifier_s"]

    logger.info("=== [%s run%d] epoch times (s) ===", dataset_label, run_index)
    for i, t in enumerate(pretrain_times, 1):
        logger.info("[pretrain] epoch %d: %.2f s", i, t)
    for i, t in enumerate(classifier_times, 1):
        logger.info("[classifier] epoch %d: %.2f s", i, t)
    logger.info(
        "pretrain_total=%.2f s, classifier_total=%.2f s, run_total=%.2f s",
        sum(pretrain_times),
        sum(classifier_times),
        sum(pretrain_times) + sum(classifier_times),
    )

    if not epoch_time_dir:
        return

    os.makedirs(epoch_time_dir, exist_ok=True)
    out_path = os.path.join(epoch_time_dir, f"{dataset_label}_run{run_index}.txt")
    lines = [
        f"# {dataset_label} run{run_index} epoch times (s)",
        "",
        "## pretrain",
    ]
    lines.extend(f"epoch {i}: {t:.2f}" for i, t in enumerate(pretrain_times, 1))
    lines += ["", "## classifier"]
    lines.extend(f"epoch {i}: {t:.2f}" for i, t in enumerate(classifier_times, 1))
    lines += [
        "",
        f"pretrain_total: {sum(pretrain_times):.2f}",
        f"classifier_total: {sum(classifier_times):.2f}",
        f"run_total: {sum(pretrain_times) + sum(classifier_times):.2f}",
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("epoch times saved: %s", out_path)


def train_barlow_once(args: Any, logger: logging.Logger) -> tuple[RunMetrics, RunEpochTimes]:
    """Barlow Twins 预训练 + 冻结 encoder 线性分类，返回 test 最佳准确率。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = args.image_size
    feat_dim = 2048

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    train_dir = os.path.join(args.data_root, "train")
    test_dir = os.path.join(args.data_root, "test")
    two_view_train = TwoViewImageFolder(train_dir, train_transform)
    train_eval_set = datasets.ImageFolder(train_dir, transform=eval_transform)
    test_set = datasets.ImageFolder(test_dir, transform=eval_transform)

    num_classes = args.num_classes if args.num_classes > 0 else len(two_view_train.base.classes)

    pretrain_loader = DataLoader(
        two_view_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    train_loader_eval = DataLoader(
        train_eval_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = _build_barlow_twins_imagenet_encoder(device, args.proj_channels)
    optimizer_pre = torch.optim.Adam(
        model.parameters(), lr=args.lr_pretrain, weight_decay=args.weight_decay
    )
    criterion_ce = nn.CrossEntropyLoss()

    base_name = build_base_name(args.data_root)
    logger.info(
        "Barlow Twins: data_root=%s, encoder_dim=%d, proj_channels=%d, lambd=%.6f, pretrain_epochs=%d",
        args.data_root,
        feat_dim,
        args.proj_channels,
        args.lambd,
        args.pretrain_epochs,
    )

    # ---------- 阶段一：Barlow Twins 自监督 ----------
    pretrain_times_s: list[float] = []
    model.train()
    for epoch in range(1, args.pretrain_epochs + 1):
        t0 = time.perf_counter()
        epoch_loss = 0.0
        n_batches = 0
        for v1, v2, _ in pretrain_loader:
            v1 = v1.to(device, non_blocking=True)
            v2 = v2.to(device, non_blocking=True)
            optimizer_pre.zero_grad()
            z1, z2 = model(v1, v2)
            loss = barlow_loss_fun(z1, z2, args.lambd)
            loss.backward()
            optimizer_pre.step()
            epoch_loss += loss.item()
            n_batches += 1
        pretrain_times_s.append(time.perf_counter() - t0)
        mean_loss = epoch_loss / max(n_batches, 1)
        if epoch == 1 or epoch % 5 == 0 or epoch == args.pretrain_epochs:
            logger.info(
                "[Barlow pretrain] epoch %d/%d loss=%.6f time=%.2fs",
                epoch,
                args.pretrain_epochs,
                mean_loss,
                pretrain_times_s[-1],
            )

    # ---------- 阶段二：冻结 BarlowTwins，仅训练线性头（encoder 2048-d）----------
    for p in model.parameters():
        p.requires_grad = False
    classifier = nn.Linear(feat_dim, num_classes).to(device)
    optimizer_cls = torch.optim.Adam(
        classifier.parameters(), lr=args.lr_classifier, weight_decay=1e-5
    )

    best_metrics = RunMetrics(
        accuracy=0.0, macro_f1=0.0, weighted_f1=0.0, balanced_accuracy=0.0
    )
    save_path = os.path.join(_ROOT, "model", f"{base_name}-best.pth")

    logger.info(
        "Linear probe: epochs=%d, classes=%d",
        args.classifier_epochs,
        num_classes,
    )

    classifier_times_s: list[float] = []
    for epoch in range(1, args.classifier_epochs + 1):
        t0 = time.perf_counter()
        epoch_loss = 0.0
        n_batches = 0
        for images, labels in train_loader_eval:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.no_grad():
                h = model.encoder(images)
            optimizer_cls.zero_grad()
            logits = classifier(h)
            loss = criterion_ce(logits, labels)
            loss.backward()
            optimizer_cls.step()
            epoch_loss += loss.item()
            n_batches += 1

        metrics = evaluate_probe_metrics(
            model, classifier, test_loader, device, num_classes
        )
        classifier_times_s.append(time.perf_counter() - t0)
        mean_loss = epoch_loss / max(n_batches, 1)
        if epoch == 1 or epoch % 5 == 0 or epoch == args.classifier_epochs:
            logger.info(
                "[Linear probe] epoch %d/%d train_loss=%.4f acc=%.4f macro_f1=%.4f "
                "weighted_f1=%.4f balanced_acc=%.4f time=%.2fs",
                epoch,
                args.classifier_epochs,
                mean_loss,
                metrics["accuracy"],
                metrics["macro_f1"],
                metrics["weighted_f1"],
                metrics["balanced_accuracy"],
                classifier_times_s[-1],
            )
        if metrics["accuracy"] > best_metrics["accuracy"]:
            best_metrics = metrics
            torch.save(
                {
                    "epoch": epoch,
                    "best_test_acc": best_metrics["accuracy"],
                    "barlow_state_dict": model.state_dict(),
                    "classifier_state_dict": classifier.state_dict(),
                    "feat_dim": feat_dim,
                    "proj_channels": args.proj_channels,
                    "lambd": args.lambd,
                    "num_classes": num_classes,
                    "class_to_idx": train_eval_set.class_to_idx,
                    "data_root": args.data_root,
                    "best_metrics": dict(best_metrics),
                },
                save_path,
            )
            logger.info(
                "Best updated: acc=%.4f -> %s",
                best_metrics["accuracy"],
                save_path,
            )

    logger.info(
        "Done. acc=%.4f macro_f1=%.4f weighted_f1=%.4f balanced_acc=%.4f",
        best_metrics["accuracy"],
        best_metrics["macro_f1"],
        best_metrics["weighted_f1"],
        best_metrics["balanced_accuracy"],
    )
    epoch_times = RunEpochTimes(
        pretrain_s=pretrain_times_s,
        classifier_s=classifier_times_s,
    )
    return best_metrics, epoch_times


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=f"{MODEL_NAME}（barlowtwins/barlow.py）+ 线性探针，五数据集风格分类"
    )
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--num_classes", type=int, default=0)
    p.add_argument("--benchmark_all", action="store_true")
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
        "--epoch_time_dir",
        type=str,
        default=None,
        help="每轮 run 的 epoch 耗时记录目录（秒）",
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
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument(
        "--pretrain_epochs",
        type=int,
        default=50,
        help="Barlow Twins 自监督轮数",
    )
    p.add_argument(
        "--classifier_epochs",
        type=int,
        default=100,
        help="冻结 encoder 后线性分类器训练轮数",
    )
    p.add_argument(
        "--proj_channels",
        type=int,
        default=2048,
        help="投影 MLP 宽度（barlow.py 中三层同宽，默认 2048 省显存；论文常用 8192）",
    )
    p.add_argument(
        "--lambd",
        type=float,
        default=0.0051,
        help="barlow_loss_fun 中非对角项权重 λ（论文默认 0.0051）",
    )
    p.add_argument(
        "--lr_pretrain",
        type=float,
        default=1e-3,
        help="自监督阶段 Adam（与 simclr_train 一致）",
    )
    p.add_argument(
        "--lr_classifier",
        type=float,
        default=1e-2,
        help="线性头 Adam（与 simclr_train 一致）",
    )
    p.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="自监督阶段权重衰减（与 simclr_train 一致）",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_runs < 1:
        raise SystemExit("错误: --run 须 >= 1")

    if not args.benchmark_all and not args.data_root:
        raise SystemExit("错误: 单数据集模式必须指定 --data_root（或使用 --benchmark_all）")

    os.environ.setdefault("TORCH_HOME", os.path.join(_ROOT, "pretrainModels"))
    os.makedirs(os.path.join(_ROOT, "model"), exist_ok=True)
    os.makedirs(os.path.join(_ROOT, "log"), exist_ok=True)
    os.makedirs(os.path.join(_ROOT, "pretrainModels"), exist_ok=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.result_md)), exist_ok=True)

    if args.benchmark_all:
        results: list[DatasetResult] = []
        data_base = os.path.normpath(args.data_base)
        for rel, n_cls, label in BENCHMARK_DATASETS:
            args.data_root = os.path.join(data_base, rel.replace("/", os.sep))
            args.num_classes = n_cls
            all_runs: list[RunMetrics] = []
            for r in range(1, args.num_runs + 1):
                log_path = os.path.join(
                    _ROOT, "log", f"{build_base_name(args.data_root)}-run{r}.log"
                )
                logger = _make_logger(log_path)
                try:
                    metrics, epoch_times = train_barlow_once(args, logger)
                    all_runs.append(metrics)
                    write_run_epoch_times(
                        args.epoch_time_dir, label, r, epoch_times, logger
                    )
                    print(
                        f"[{label} run{r}/{args.num_runs}] "
                        f"acc={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}, "
                        f"weighted_f1={metrics['weighted_f1']:.4f}, "
                        f"balanced_acc={metrics['balanced_accuracy']:.4f}"
                    )
                except Exception as e:
                    logger.exception("Dataset failed: %s run %d", label, r)
                    all_runs.append(
                        RunMetrics(
                            accuracy=float("nan"),
                            macro_f1=float("nan"),
                            weighted_f1=float("nan"),
                            balanced_accuracy=float("nan"),
                        )
                    )
                    print(f"[{label} run{r}/{args.num_runs}] FAILED: {e}")
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
            args.pretrain_epochs,
            args.classifier_epochs,
            args.num_runs,
        )
        print(f"结果已写入: {args.result_md}")
        return

    data_root = os.path.abspath(args.data_root.rstrip("/"))
    dataset_name = args.dataset_label or os.path.basename(os.path.normpath(data_root))
    all_runs: list[RunMetrics] = []
    for r in range(1, args.num_runs + 1):
        log_path = os.path.join(
            _ROOT, "log", f"{build_base_name(args.data_root)}-run{r}.log"
        )
        logger = _make_logger(log_path)
        print(f"[{dataset_name} run{r}/{args.num_runs}] 开始训练…")
        try:
            metrics, epoch_times = train_barlow_once(args, logger)
            all_runs.append(metrics)
            write_run_epoch_times(
                args.epoch_time_dir, dataset_name, r, epoch_times, logger
            )
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
            args.pretrain_epochs,
            args.classifier_epochs,
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
