"""
MCCFNet：端到端训练与测试集评估；支持单数据集或六数据集 benchmark（与 selfsupervised 六库约定一致）。
"""
from __future__ import annotations

import argparse
import logging
import os
import shlex
import sys
import time
from argparse import Namespace

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import DenseNet169_Weights, densenet169

# 与 selfsupervised/barlowtwins_train.py 中 BENCHMARK_DATASETS 一致
BENCHMARK_DATASETS: list[tuple[str, int, str]] = [
    ("Painting91", 13, "Painting91"),
    ("Pandora", 12, "Pandora"),
    ("AVAstyle", 14, "AVAstyle"),
    ("FashionStyle14", 14, "FashionStyle14"),
    ("Arch", 25, "Arch"),
    ("webstyle", 10, "WebStyle"),
]


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
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return correct / total if total > 0 else 0.0


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


def setup_torch_hub(root: str) -> None:
    hub_dir = os.path.join(root, "pretrainModels", "hub")
    os.makedirs(hub_dir, exist_ok=True)
    torch.hub.set_dir(hub_dir)
    os.environ.setdefault("TORCH_HOME", os.path.join(root, "pretrainModels"))


def append_benchmark_markdown(
    result_path: str,
    rows: list[tuple[str, int, float, str]],
    data_base: str,
    argv_summary: str,
    epochs: int,
) -> None:
    """rows: (label, num_classes, best_test_acc, data_root)"""
    os.makedirs(os.path.dirname(os.path.abspath(result_path)), exist_ok=True)
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    lines = [
        "",
        f"## MCCFNet 六数据集 (DenseNet169+RWP, epochs={epochs}) — {ts}",
        "",
        f"_data_base=`{data_base}`_",
        "",
        f"_命令: `{argv_summary}`_",
        "",
        "| Dataset | num_classes | best_test_acc | data_root |",
        "|---------|-------------|---------------|-----------|",
    ]
    for name, nc, acc, droot in rows:
        acc_s = "FAILED" if acc != acc else f"{acc:.4f}"
        lines.append(f"| {name} | {nc} | {acc_s} | `{droot}` |")
    lines.append("")

    with open(result_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))


def train_single_dataset(args: Namespace, logger: logging.Logger) -> float:
    """
    训练单个数据集上的 MCCFNet，返回测试集最佳准确率。
    端到端训练；线性分类头与骨干同时优化，「迭代」即外层 epoch（默认 20）。
    """
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

    best_acc = 0.0
    best_path = os.path.join(args.save_dir, f"{base_name}-best.pth")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_acc = evaluate(model, test_loader, device)
        logger.info(
            "Epoch [%d/%d] train_loss=%.4f test_acc=%.4f",
            epoch,
            args.epochs,
            train_loss,
            test_acc,
        )
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "test_acc": test_acc,
                    "num_classes": num_classes,
                    "class_to_idx": train_set.class_to_idx,
                    "args": vars(args),
                },
                best_path,
            )
            logger.info("Saved best checkpoint: %s (test_acc=%.4f)", best_path, test_acc)

    logger.info("Done. Best test accuracy: %.4f", best_acc)
    print(f"Best test accuracy: {best_acc:.4f}  (checkpoint: {best_path})")
    return best_acc


def parse_args() -> Namespace:
    p = argparse.ArgumentParser(description="MCCFNet 分类训练（单数据集或六数据集 benchmark）")
    p.add_argument(
        "--data_root",
        type=str,
        default="/mnt/codes/data/style/Painting91",
        help="数据集根目录（含 train/test）；benchmark 模式下由 data_base+子目录覆盖",
    )
    p.add_argument("--num_classes", type=int, default=13, help="<=0 时从 train 推断")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="训练轮数（端到端；含线性分类头）。六数据集批量默认 20。",
    )
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--save_dir", type=str, default="model")
    p.add_argument(
        "--benchmark_all",
        action="store_true",
        help="依次在六数据集上训练并测试（data_base 下子目录同 barlowtwins/simclr）",
    )
    p.add_argument(
        "--data_base",
        type=str,
        default="/mnt/codes/data/style/",
        help="benchmark 时数据根（各库为其子目录）",
    )
    p.add_argument(
        "--result_md",
        type=str,
        default="",
        help="benchmark 结果追加写入的 Markdown 路径；默认 MCCFNet/mccfnet_result.md",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(root)
    setup_torch_hub(root)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs("log", exist_ok=True)

    result_md = args.result_md or os.path.join(root, "MCCFNet", "mccfnet_result.md")

    if args.benchmark_all:
        argv_summary = shlex.join([sys.argv[0]] + sys.argv[1:])
        rows: list[tuple[str, int, float, str]] = []
        for rel, n_cls, label in BENCHMARK_DATASETS:
            args.data_root = os.path.join(os.path.normpath(args.data_base), rel.replace("/", os.sep))
            args.num_classes = n_cls
            dataset_name = os.path.basename(os.path.normpath(args.data_root))
            time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            log_path = os.path.join(
                "log", f"mccfnet-benchmark-{dataset_name}-{time_str}.log"
            )
            logger = logging.getLogger(f"mccfnet.{label}")
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
            try:
                best = train_single_dataset(args, logger)
                rows.append((label, n_cls, best, os.path.abspath(args.data_root)))
                print(f"[{label}] best_test_acc={best:.4f}")
            except Exception as e:
                logger.exception("Dataset failed: %s", label)
                rows.append((label, n_cls, float("nan"), os.path.abspath(args.data_root)))
                print(f"[{label}] FAILED: {e}")

        append_benchmark_markdown(
            result_md,
            rows,
            os.path.normpath(args.data_base) + os.sep,
            argv_summary,
            args.epochs,
        )
        print(f"结果已追加写入: {result_md}")
        return

    # 单数据集
    time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    dataset_name = os.path.basename(os.path.normpath(args.data_root))
    log_path = os.path.join("log", f"mccfnet-densenet169-{dataset_name}-{time_str}.log")
    logger = logging.getLogger("mccfnet_train")
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
    logger.info("Log: %s", log_path)
    train_single_dataset(args, logger)


if __name__ == "__main__":
    main()
