"""
基于 simclr/simclr.py 中的 SimCLR（NT-Xent）与 ResNet50 骨干：
1) 双视图对比预训练；2) 冻结骨干，训练线性分类头并在 test 上报告准确率。

六数据集批量与 denoise/dae_train.py 对齐，结果追加写入 selfsupervised/simclr_result.md。

用法::
    python selfsupervised/simclr_train.py --benchmark_all
    python selfsupervised/simclr_train.py --data_root /path/to/Painting91 --num_classes 13
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from simclr.simclr import SimCLR  # noqa: E402

BENCHMARK_DATASETS: list[tuple[str, int, str]] = [
    ("Painting91", 13, "Painting91"),
    ("Pandora", 12, "Pandora"),
    ("AVAstyle", 14, "AVAstyle"),
    ("FashionStyle14", 14, "FashionStyle14"),
    ("Arch", 25, "Arch"),
    ("webstyle", 10, "WebStyle"),
]

DEFAULT_RESULT_MD = os.path.join(_ROOT, "selfsupervised", "simclr_result.md")


def _identity_fc_resnet50(device: torch.device) -> tuple[nn.Module, int]:
    """
    ImageNet 预训练 ResNet50，将 fc 换为恒等映射（权重为单位阵），便于 SimCLR 的 projection_MLP 使用 fc.out_features。
    """
    try:
        w = models.ResNet50_Weights.IMAGENET1K_V1
        backbone = models.resnet50(weights=w)
    except Exception:
        backbone = models.resnet50(pretrained=True)
    d = backbone.fc.in_features
    lin = nn.Linear(d, d)
    with torch.no_grad():
        lin.weight.copy_(torch.eye(d))
        lin.bias.zero_()
    backbone.fc = lin
    return backbone.to(device), d


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
    return f"simclr-resnet50-{dataset_name}-{time_str}"


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
def evaluate_linear(
    simclr: nn.Module,
    classifier: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    simclr.eval()
    classifier.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        h = simclr.backbone(images)
        logits = classifier(h)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return correct / total if total > 0 else 0.0


def append_benchmark_markdown(
    result_path: str,
    rows: list[tuple[str, int, list[float], float, float, str]],
    data_base: str,
    argv_summary: str,
    num_runs: int,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(result_path)), exist_ok=True)
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    run_headers = [f"run{i}" for i in range(1, num_runs + 1)]
    header = "| Dataset | num_classes | " + " | ".join(run_headers) + " | mean±std | data_root |"
    n_cols = 4 + num_runs
    sep = "|" + "|".join(["---------"] * n_cols) + "|"

    lines = [
        "",
        f"## SimCLR 六数据集 (backbone=resnet50, run={num_runs}) — {ts}",
        "",
        f"_data_base=`{data_base}`_",
        "",
        f"_命令: `{argv_summary}`_",
        "",
        header,
        sep,
    ]
    for name, nc, accs, mean_v, std_v, droot in rows:
        acc_cells: list[str] = []
        for i in range(num_runs):
            if i < len(accs):
                v = accs[i]
                acc_cells.append("nan" if v != v else f"{v:.4f}")
            else:
                acc_cells.append("-")
        arr = np.asarray(accs, dtype=np.float64)
        if accs and np.all(np.isnan(arr)):
            mean_std = "FAILED"
        else:
            mean_std = f"{mean_v:.4f}±{std_v:.4f}"
        lines.append(
            f"| {name} | {nc} | "
            + " | ".join(acc_cells)
            + f" | {mean_std} | `{droot}` |"
        )
    lines.append("")

    with open(result_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))


def train_simclr_once(args: Any, logger: logging.Logger) -> float:
    """SimCLR 预训练 + 冻结骨干线性分类，返回 test 最佳准确率。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = args.image_size

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

    backbone, feat_dim = _identity_fc_resnet50(device)
    model = SimCLR(backbone=backbone).to(device)

    optimizer_pre = torch.optim.Adam(
        model.parameters(), lr=args.lr_pretrain, weight_decay=args.weight_decay
    )
    criterion_ce = nn.CrossEntropyLoss()

    base_name = build_base_name(args.data_root)
    logger.info(
        "SimCLR pretrain: data_root=%s, feats=%d, pretrain_epochs=%d, image_size=%d",
        args.data_root,
        feat_dim,
        args.pretrain_epochs,
        image_size,
    )

    # ---------- 阶段一：SimCLR 对比预训练 ----------
    model.train()
    for epoch in range(1, args.pretrain_epochs + 1):
        epoch_loss = 0.0
        n_batches = 0
        for v1, v2, _ in pretrain_loader:
            v1 = v1.to(device, non_blocking=True)
            v2 = v2.to(device, non_blocking=True)
            optimizer_pre.zero_grad()
            out = model(v1, v2)
            loss = out["loss"].mean()
            loss.backward()
            optimizer_pre.step()
            epoch_loss += loss.item()
            n_batches += 1
        mean_loss = epoch_loss / max(n_batches, 1)
        if epoch == 1 or epoch % 5 == 0 or epoch == args.pretrain_epochs:
            logger.info(
                "[SimCLR pretrain] epoch %d/%d loss=%.6f",
                epoch,
                args.pretrain_epochs,
                mean_loss,
            )

    # ---------- 阶段二：冻结骨干，训练线性头 ----------
    for p in model.parameters():
        p.requires_grad = False
    classifier = nn.Linear(feat_dim, num_classes).to(device)
    optimizer_cls = torch.optim.Adam(
        classifier.parameters(), lr=args.lr_classifier, weight_decay=1e-5
    )

    best_acc = 0.0
    save_path = os.path.join(_ROOT, "model", f"{base_name}-best.pth")

    logger.info(
        "Linear probe: epochs=%d, classes=%d",
        args.classifier_epochs,
        num_classes,
    )

    for epoch in range(1, args.classifier_epochs + 1):
        epoch_loss = 0.0
        n_batches = 0
        for images, labels in train_loader_eval:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.no_grad():
                h = model.backbone(images)
            optimizer_cls.zero_grad()
            logits = classifier(h)
            loss = criterion_ce(logits, labels)
            loss.backward()
            optimizer_cls.step()
            epoch_loss += loss.item()
            n_batches += 1

        test_acc = evaluate_linear(model, classifier, test_loader, device)
        mean_loss = epoch_loss / max(n_batches, 1)
        if epoch == 1 or epoch % 5 == 0 or epoch == args.classifier_epochs:
            logger.info(
                "[Linear probe] epoch %d/%d train_loss=%.4f test_acc=%.4f",
                epoch,
                args.classifier_epochs,
                mean_loss,
                test_acc,
            )
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(
                {
                    "epoch": epoch,
                    "best_test_acc": best_acc,
                    "simclr_state_dict": model.state_dict(),
                    "classifier_state_dict": classifier.state_dict(),
                    "feat_dim": feat_dim,
                    "num_classes": num_classes,
                    "class_to_idx": train_eval_set.class_to_idx,
                    "data_root": args.data_root,
                },
                save_path,
            )
            logger.info("Best updated: test_acc=%.4f -> %s", best_acc, save_path)

    logger.info("Done. best_test_acc=%.4f", best_acc)
    return best_acc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SimCLR（simclr/simclr.py）+ 线性探针，六数据集或单数据集风格分类"
    )
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--num_classes", type=int, default=0)
    p.add_argument("--benchmark_all", action="store_true")
    p.add_argument("--data_base", type=str, default="/mnt/codes/data/style/")
    p.add_argument("--result_md", type=str, default=DEFAULT_RESULT_MD)
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
    p.add_argument("--pretrain_epochs", type=int, default=100, help="SimCLR 对比学习轮数")
    p.add_argument(
        "--classifier_epochs",
        type=int,
        default=100,
        help="冻结骨干后线性分类器训练轮数",
    )
    p.add_argument("--lr_pretrain", type=float, default=1e-3, help="SimCLR 阶段 Adam")
    p.add_argument("--lr_classifier", type=float, default=1e-2, help="线性头 Adam")
    p.add_argument("--weight_decay", type=float, default=1e-5)
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

    if args.benchmark_all:
        rows: list[tuple[str, int, list[float], float, float, str]] = []
        import shlex

        argv_summary = shlex.join([sys.argv[0]] + sys.argv[1:])
        for rel, n_cls, label in BENCHMARK_DATASETS:
            args.data_root = os.path.join(
                os.path.normpath(args.data_base), rel.replace("/", os.sep)
            )
            args.num_classes = n_cls
            accs: list[float] = []
            for r in range(1, args.num_runs + 1):
                log_path = os.path.join(
                    _ROOT,
                    "log",
                    f"{build_base_name(args.data_root)}-run{r}.log",
                )
                logger = _make_logger(log_path)
                try:
                    best = train_simclr_once(args, logger)
                    accs.append(best)
                    print(f"[{label} run{r}/{args.num_runs}] best_test_acc={best:.4f}")
                except Exception as e:
                    logger.exception("Dataset failed: %s run %d", label, r)
                    accs.append(float("nan"))
                    print(f"[{label} run{r}/{args.num_runs}] FAILED: {e}")

            arr = np.asarray(accs, dtype=np.float64)
            mean_v = float(np.nanmean(arr))
            std_v = (
                float(np.nanstd(arr, ddof=1))
                if np.sum(~np.isnan(arr)) > 1
                else 0.0
            )
            rows.append(
                (label, n_cls, accs, mean_v, std_v, os.path.abspath(args.data_root))
            )
            print(f"[{label}] mean±std = {mean_v:.4f}±{std_v:.4f}")

        append_benchmark_markdown(
            args.result_md,
            rows,
            os.path.normpath(args.data_base) + os.sep,
            argv_summary,
            args.num_runs,
        )
        print(f"结果已追加写入: {args.result_md}")
        return

    accs_single: list[float] = []
    for r in range(1, args.num_runs + 1):
        log_path = os.path.join(
            _ROOT, "log", f"{build_base_name(args.data_root)}-run{r}.log"
        )
        logger = _make_logger(log_path)
        best_acc = train_simclr_once(args, logger)
        accs_single.append(best_acc)
        print(f"[run {r}/{args.num_runs}] best_test_acc={best_acc:.4f}")
    arr_s = np.asarray(accs_single, dtype=np.float64)
    mean_s = float(np.nanmean(arr_s))
    std_s = float(np.nanstd(arr_s, ddof=1)) if np.sum(~np.isnan(arr_s)) > 1 else 0.0
    print(f"mean±std: {mean_s:.4f}±{std_s:.4f}")


if __name__ == "__main__":
    main()
