"""
冻结 VGG16 分类器前一层（4096 维）特征训练 CSCAE，在 test 上报告最佳准确率。
（与 traditional_train 中 vgg16 特征定义一致，可通过 --backbone 切换其他主干。）

单数据集::
    python denoise/sscae_train.py --data_root /path/to/Painting91 --num_classes 13

六数据集批量（与项目 remote_sh 中路径/类别约定一致）::
    python denoise/sscae_train.py --benchmark_all --data_base /mnt/codes/data/style/

每个数据集默认独立重复训练 3 次（--runs），结果文件记录各次 best_test_acc 及 mean±std。
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
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from traditional_train import BACKBONE_CONFIGS, build_backbone, build_feature_cache  # noqa: E402

from denoise.SSCAE import CSCAE, CSCAELoss  # noqa: E402

# (相对 data_base 的子路径, 类别数, Markdown 表中的名称)
BENCHMARK_DATASETS: list[tuple[str, int, str]] = [
    ("Painting91", 13, "Painting91"),
    ("Pandora", 12, "Pandora"),
    ("AVAstyle", 14, "AVAstyle"),
    ("FashionStyle14", 14, "FashionStyle14"),
    ("Arch", 25, "Arch"),
    ("webstyle", 10, "WebStyle"),
]

DEFAULT_RESULT_MD = os.path.join(_ROOT, "denoise", "sscae_result.md")


def train_step(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad()
    latents, reconstructions, consensus_latent, logits = model(x)
    loss, loss_dict = criterion(
        x, labels, latents, reconstructions, consensus_latent, model.centers, logits
    )
    loss.backward()
    optimizer.step()
    return loss_dict


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    correct = 0
    total = 0
    for feats, labels in test_loader:
        feats = feats.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        _, _, _, logits = model(feats)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return correct / total if total > 0 else 0.0


def build_base_name(data_root: str, backbone: str) -> str:
    dataset_name = os.path.basename(os.path.normpath(data_root))
    time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    return f"sscae-{backbone}-{dataset_name}-{time_str}"


def _make_logger(log_path: str) -> logging.Logger:
    """每个任务独立 logger，避免多数据集连续跑时 handler 叠加。"""
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


def train_cscae(args: Any, logger: logging.Logger) -> float:
    """
    使用 args.data_root、args.num_classes 及其余超参完成一次训练，返回 test 最佳准确率。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_dim, input_size = BACKBONE_CONFIGS[args.backbone]

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

    input_dim = feat_dim
    model = CSCAE(
        args.num_ensembles, input_dim, args.hidden_dim, args.latent_dim, num_classes
    ).to(device)
    logger.info("\n%s", model.describe_feature_dims())
    criterion = CSCAELoss(
        lambda_recon=args.lambda_recon,
        lambda_center=args.lambda_center,
        lambda_consensus=args.lambda_consensus,
    )
    optimizer = torch.optim.Adam(
        [
            {"params": model.autoencoders.parameters()},
            {"params": model.classifier.parameters()},
            {"params": [model.centers], "lr": args.center_lr},
        ],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_acc = 0.0
    save_path = os.path.join(_ROOT, "model", f"{base_name}-best.pth")

    logger.info(
        "Start CSCAE: data_root=%s, classes=%d, K=%d, latent=%d, λ=(recon=%.3f, center=%.3f, consensus=%.3f)",
        args.data_root,
        num_classes,
        args.num_ensembles,
        args.latent_dim,
        args.lambda_recon,
        args.lambda_center,
        args.lambda_consensus,
    )

    for epoch in range(1, args.epochs + 1):
        epoch_losses: dict[str, list[float]] = {
            "ce_loss": [],
            "recon_loss": [],
            "center_loss": [],
            "consensus_loss": [],
        }
        for feats, labels in train_loader:
            feats = feats.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            loss_dict = train_step(model, criterion, optimizer, feats, labels, device)
            for k in epoch_losses:
                epoch_losses[k].append(loss_dict[k])

        test_acc = evaluate(model, test_loader, device)
        mean_ce = float(np.mean(epoch_losses["ce_loss"]))
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            logger.info(
                "[Epoch %3d/%d] ce=%.4f | test_acc=%.4f",
                epoch,
                args.epochs,
                mean_ce,
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
                    "class_to_idx": train_set.class_to_idx,
                    "num_classes": num_classes,
                    "data_root": args.data_root,
                    "hparams": {
                        "input_dim": input_dim,
                        "hidden_dim": args.hidden_dim,
                        "latent_dim": args.latent_dim,
                        "num_ensembles": args.num_ensembles,
                    },
                },
                save_path,
            )
            logger.info("Best updated: test_acc=%.4f -> %s", best_acc, save_path)

    logger.info("Training done. best_test_acc=%.4f", best_acc)
    return best_acc


def append_benchmark_markdown(
    result_path: str,
    rows: list[tuple[str, int, list[float], float, float, str]],
    data_base: str,
    argv_summary: str,
    backbone: str,
    num_runs: int,
) -> None:
    """
    rows: (显示名, num_classes, 各次 best_test_acc 列表, mean, std, data_root 绝对路径)
    """
    os.makedirs(os.path.dirname(os.path.abspath(result_path)), exist_ok=True)
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    run_headers = [f"run{i}" for i in range(1, num_runs + 1)]
    header = "| Dataset | num_classes | " + " | ".join(run_headers) + " | mean±std | data_root |"
    n_cols = 4 + num_runs
    sep = "|" + "|".join(["---------"] * n_cols) + "|"

    lines = [
        "",
        f"## SSCAE 六数据集 (backbone={backbone}, runs={num_runs}) — {ts}",
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
                acc_cells.append("nan" if v != v else f"{v:.4f}")  # nan != nan
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CSCAE + frozen backbone features（默认 VGG16）")
    p.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="数据集根目录（含 train/、test/）；与 --benchmark_all 互斥",
    )
    p.add_argument(
        "--num_classes",
        type=int,
        default=0,
        help="类别数；<=0 时从 train 目录推断（单数据集模式）",
    )
    p.add_argument(
        "--benchmark_all",
        action="store_true",
        help="依次在 Painting91、Pandora、AVAstyle、FashionStyle14、Arch、WebStyle 上训练并汇总到结果文件",
    )
    p.add_argument(
        "--data_base",
        type=str,
        default="/mnt/codes/data/style/",
        help="--benchmark_all 时与 BENCHMARK_DATASETS 中的相对路径拼接",
    )
    p.add_argument(
        "--result_md",
        type=str,
        default=DEFAULT_RESULT_MD,
        help=f"批量模式结束时追加写入的 Markdown 路径（默认: {DEFAULT_RESULT_MD}）",
    )
    p.add_argument(
        "--run",
        "--runs",
        type=int,
        default=3,
        dest="runs",
        metavar="N",
        help="每个数据集（或单数据集模式下的同一数据）独立重复训练次数，记录各次 best 并汇总 mean±std（--run 与 --runs 等价）",
    )
    p.add_argument(
        "--backbone",
        type=str,
        default="vgg16",
        choices=list(BACKBONE_CONFIGS),
        help="冻结特征主干（默认 vgg16，输出 4096 维；resnet50 为 2048 维 GAP）",
    )
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--hidden_dim", type=int, default=1024)
    p.add_argument("--latent_dim", type=int, default=256)
    p.add_argument("--num_ensembles", type=int, default=3)
    p.add_argument("--lambda_recon", type=float, default=1.0)
    p.add_argument("--lambda_center", type=float, default=0.01)
    p.add_argument("--lambda_consensus", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--center_lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.runs < 1:
        raise SystemExit("错误: --run / --runs 须 >= 1")

    if args.benchmark_all:
        data_base = os.path.normpath(args.data_base) + os.sep
    else:
        if not args.data_root:
            raise SystemExit("错误: 单数据集模式必须指定 --data_root（或使用 --benchmark_all）")
        data_base = ""

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
            for r in range(1, args.runs + 1):
                log_path = os.path.join(
                    _ROOT,
                    "log",
                    f"{build_base_name(args.data_root, args.backbone)}-run{r}.log",
                )
                logger = _make_logger(log_path)
                try:
                    best = train_cscae(args, logger)
                    accs.append(best)
                    print(f"[{label} run{r}/{args.runs}] best_test_acc={best:.4f}")
                except Exception as e:
                    logger.exception("Dataset failed: %s run %d", label, r)
                    accs.append(float("nan"))
                    print(f"[{label} run{r}/{args.runs}] FAILED: {e}")

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
            args.runs,
        )
        print(f"结果已追加写入: {args.result_md}")
        return

    # 单数据集：重复 --runs 次
    accs_single: list[float] = []
    for r in range(1, args.runs + 1):
        log_path = os.path.join(
            _ROOT, "log", f"{build_base_name(args.data_root, args.backbone)}-run{r}.log"
        )
        logger = _make_logger(log_path)
        best_acc = train_cscae(args, logger)
        accs_single.append(best_acc)
        print(f"[run {r}/{args.runs}] best_test_acc={best_acc:.4f}")
    arr_s = np.asarray(accs_single, dtype=np.float64)
    mean_s = float(np.nanmean(arr_s))
    if np.sum(~np.isnan(arr_s)) > 1:
        std_s = float(np.nanstd(arr_s, ddof=1))
    else:
        std_s = 0.0
    print(f"mean±std: {mean_s:.4f}±{std_s:.4f}")


if __name__ == "__main__":
    main()
