"""
堆叠降噪自编码器 SDAEClassifier（见 denoise/DAE.py）：**固定堆叠 2 个 DAE 单元**；冻结主干特征 → 逐层 DAE 预训练 → 有监督微调。
流程与 denoise/sscae_train.py 对齐：六数据集批量、--run 多轮训练、结果 Markdown。

单数据集::
    python denoise/dae_train.py --data_root /path/to/Painting91 --num_classes 13

六数据集（默认每种数据训练 3 轮，可用 ``--run 5`` 改为 5 轮）::
    python denoise/dae_train.py --benchmark_all --data_base /mnt/codes/data/style/
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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from traditional_train import BACKBONE_CONFIGS, build_backbone, build_feature_cache  # noqa: E402

from denoise.DAE import SDAEClassifier, layer_dims_two_stacked_dae  # noqa: E402

BENCHMARK_DATASETS: list[tuple[str, int, str]] = [
    ("Painting91", 13, "Painting91"),
    ("Pandora", 12, "Pandora"),
    ("AVAstyle", 14, "AVAstyle"),
    ("FashionStyle14", 14, "FashionStyle14"),
    ("Arch", 25, "Arch"),
    ("webstyle", 10, "WebStyle"),
]

DEFAULT_RESULT_MD = os.path.join(_ROOT, "denoise", "dae_result.md")


def parse_hidden_dims(feat_dim: int, spec: str) -> list[int]:
    """
    固定堆叠 **2 个** DAE 单元：需且仅需两个隐层维度 h1,h2。
    例如 feat_dim=4096, spec='1024,512' -> [4096, 1024, 512]（见 layer_dims_two_stacked_dae）。
    """
    parts = [int(x.strip()) for x in spec.split(",") if x.strip()]
    if len(parts) != 2:
        raise ValueError(
            "SDAE 固定堆叠 2 个 DAE 单元，请提供两个隐层维度，例如 --hidden_dims 1024,512"
        )
    return layer_dims_two_stacked_dae(feat_dim, parts[0], parts[1])


def build_base_name(data_root: str, backbone: str) -> str:
    dataset_name = os.path.basename(os.path.normpath(data_root))
    time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    return f"sdae-{backbone}-{dataset_name}-{time_str}"


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
def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    correct = 0
    total = 0
    for feats, labels in test_loader:
        feats = feats.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(feats)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return correct / total if total > 0 else 0.0


def append_benchmark_markdown(
    result_path: str,
    rows: list[tuple[str, int, list[float], float, float, str]],
    data_base: str,
    argv_summary: str,
    backbone: str,
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
        f"## SDAE 六数据集 (backbone={backbone}, run={num_runs}) — {ts}",
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


def train_sdae_once(args: Any, logger: logging.Logger) -> float:
    """一次完整预训练 + 微调，返回 test 集最佳分类准确率。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_dim, input_size = BACKBONE_CONFIGS[args.backbone]
    layer_dims = parse_hidden_dims(feat_dim, args.hidden_dims)

    base_name = build_base_name(args.data_root, args.backbone)
    transform = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dir = os.path.join(args.data_root, "train")
    test_dir = os.path.join(args.data_root, "test")
    train_set = datasets.ImageFolder(train_dir, transform=transform)
    test_set = datasets.ImageFolder(test_dir, transform=transform)

    train_image_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_image_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    num_classes = args.num_classes if args.num_classes > 0 else len(train_set.classes)
    backbone = build_backbone(args.backbone, device)
    logger.info(
        "Extracting train/test features with frozen %s (feat_dim=%d)...",
        args.backbone,
        feat_dim,
    )
    train_cache = build_feature_cache(backbone, train_image_loader, device)
    test_cache = build_feature_cache(backbone, test_image_loader, device)
    logger.info("Feature cache: train=%d, test=%d", len(train_cache), len(test_cache))

    train_loader = DataLoader(
        train_cache,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_cache,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = SDAEClassifier(
        layer_dims,
        num_classes,
        corruption_rate=args.corruption_rate,
        continuous_features=True,
    ).to(device)

    logger.info(
        "SDAE layer_dims=%s, classes=%d, corruption=%.2f, continuous_features=True",
        layer_dims,
        num_classes,
        args.corruption_rate,
    )

    # ---------- 逐层无监督预训练 ----------
    for i, dae in enumerate(model.dae_layers):
        optimizer = torch.optim.Adam(dae.parameters(), lr=args.lr_pretrain, weight_decay=args.weight_decay)
        for epoch in range(1, args.epochs_pretrain + 1):
            dae.train()
            epoch_loss = 0.0
            n_batches = 0
            for feats, _ in train_loader:
                feats = feats.to(device, non_blocking=True)
                with torch.no_grad():
                    h = feats
                    for j in range(i):
                        h = model.dae_layers[j](h)
                optimizer.zero_grad()
                reconstructed, _ = dae.pretrain_forward(h)
                loss = F.mse_loss(reconstructed, h)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            mean_loss = epoch_loss / max(n_batches, 1)
            if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs_pretrain:
                logger.info(
                    "[Pretrain layer %d/%d] epoch %d/%d mse=%.6f",
                    i + 1,
                    len(model.dae_layers),
                    epoch,
                    args.epochs_pretrain,
                    mean_loss,
                )

    # ---------- 有监督微调（按 epoch 在 test 上取最佳准确率）----------
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr_finetune, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    save_path = os.path.join(_ROOT, "model", f"{base_name}-best.pth")

    for epoch in range(1, args.epochs_finetune + 1):
        model.train()
        for feats, labels in train_loader:
            feats = feats.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(feats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        test_acc = evaluate(model, test_loader, device)
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs_finetune:
            logger.info(
                "[Finetune] epoch %d/%d test_acc=%.4f",
                epoch,
                args.epochs_finetune,
                test_acc,
            )
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(
                {
                    "epoch": epoch,
                    "best_test_acc": best_acc,
                    "model_state_dict": model.state_dict(),
                    "backbone": args.backbone,
                    "layer_dims": layer_dims,
                    "class_to_idx": train_set.class_to_idx,
                    "num_classes": num_classes,
                    "data_root": args.data_root,
                },
                save_path,
            )
            logger.info("Best updated: test_acc=%.4f -> %s", best_acc, save_path)

    logger.info("Training done. best_test_acc=%.4f", best_acc)
    return best_acc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SDAE + frozen backbone（默认 VGG16 4096 维特征）")
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
        help="每个数据集（或单数据集）独立重复训练次数，统计各次 best 与 mean±std（默认 3）；与 --runs 等价",
    )
    p.add_argument(
        "--backbone",
        type=str,
        default="vgg16",
        choices=list(BACKBONE_CONFIGS),
    )
    p.add_argument(
        "--hidden_dims",
        type=str,
        default="1024,512",
        help="堆叠 2 个 DAE 的两层隐层维度 h1,h2（逗号分隔）；拼接后为 [feat,h1,h2]",
    )
    p.add_argument("--corruption_rate", type=float, default=0.2)
    p.add_argument("--epochs_pretrain", type=int, default=10, help="每层 DAE 无监督预训练轮数")
    p.add_argument("--epochs_finetune", type=int, default=50, help="端到端微调轮数")
    p.add_argument("--lr_pretrain", type=float, default=1e-3)
    p.add_argument("--lr_finetune", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_runs < 1:
        raise SystemExit("错误: --run 须 >= 1")

    if args.benchmark_all:
        pass
    else:
        if not args.data_root:
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
            args.data_root = os.path.join(os.path.normpath(args.data_base), rel.replace("/", os.sep))
            args.num_classes = n_cls
            accs: list[float] = []
            for r in range(1, args.num_runs + 1):
                log_path = os.path.join(
                    _ROOT,
                    "log",
                    f"{build_base_name(args.data_root, args.backbone)}-run{r}.log",
                )
                logger = _make_logger(log_path)
                try:
                    best = train_sdae_once(args, logger)
                    accs.append(best)
                    print(f"[{label} run{r}/{args.num_runs}] best_test_acc={best:.4f}")
                except Exception as e:
                    logger.exception("Dataset failed: %s run %d", label, r)
                    accs.append(float("nan"))
                    print(f"[{label} run{r}/{args.num_runs}] FAILED: {e}")

            arr = np.asarray(accs, dtype=np.float64)
            mean_v = float(np.nanmean(arr))
            if np.sum(~np.isnan(arr)) > 1:
                std_v = float(np.nanstd(arr, ddof=1))
            else:
                std_v = 0.0
            rows.append(
                (label, n_cls, accs, mean_v, std_v, os.path.abspath(args.data_root))
            )
            print(f"[{label}] mean±std = {mean_v:.4f}±{std_v:.4f}")

        append_benchmark_markdown(
            args.result_md,
            rows,
            os.path.normpath(args.data_base) + os.sep,
            argv_summary,
            args.backbone,
            args.num_runs,
        )
        print(f"结果已追加写入: {args.result_md}")
        return

    accs_single: list[float] = []
    for r in range(1, args.num_runs + 1):
        log_path = os.path.join(
            _ROOT, "log", f"{build_base_name(args.data_root, args.backbone)}-run{r}.log"
        )
        logger = _make_logger(log_path)
        best_acc = train_sdae_once(args, logger)
        accs_single.append(best_acc)
        print(f"[run {r}/{args.num_runs}] best_test_acc={best_acc:.4f}")
    arr_s = np.asarray(accs_single, dtype=np.float64)
    mean_s = float(np.nanmean(arr_s))
    if np.sum(~np.isnan(arr_s)) > 1:
        std_s = float(np.nanstd(arr_s, ddof=1))
    else:
        std_s = 0.0
    print(f"mean±std: {mean_s:.4f}±{std_s:.4f}")


if __name__ == "__main__":
    main()
