# AdaIN decoder 训练 + 线性分类评估（五数据集 benchmark）
# 用法:
#   python train.py --data_root /mnt/codes/data/style/Painting91 --num_classes 13 --runs 3
#   ./ST-SACLF-ncc_main/pytorch-AdaIN/run_st_saclf_train_bat.sh

import argparse
import copy
import glob
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, TypedDict

_SCRIPT_DIR = Path(__file__).resolve().parent
_ST_SACLF_ROOT = _SCRIPT_DIR.parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[1]
_DEFAULT_VGG = _ST_SACLF_ROOT / "models" / "vgg_normalised.pth"
os.environ.setdefault("VGG_NORMALISED_PATH", str(_DEFAULT_VGG))
os.chdir(_SCRIPT_DIR)

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image, ImageFile
from sklearn.metrics import balanced_accuracy_score, f1_score
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

import net
from sampler import InfiniteSamplerWrapper

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

MODEL_NAME = "ST-SACLF (AdaIN)"
DEFAULT_DATA_BASE = "/mnt/codes/data/style"
DEFAULT_DATA_ROOT = f"{DEFAULT_DATA_BASE}/Painting91"
NUM_CLASSES = 13

BENCHMARK_DATASETS = [
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

DEFAULT_RESULT_MD = str(_PROJECT_ROOT / "ieee_access_paperdata" / "ST-SACLF_multiple.md")

DATASET_ORDER = ["Painting91", "Pandora", "ArtBench", "FashionStyle14", "Arch"]
DATASET_REL = {
    "Painting91": "Painting91",
    "Pandora": "Pandora",
    "ArtBench": "Artbench",
    "FashionStyle14": "FashionStyle14",
    "Arch": "Arch",
}

_DECODER_INIT_STATE: Optional[dict] = None


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
    all_runs: List[RunMetrics]


def _get_decoder_init_state() -> dict:
    global _DECODER_INIT_STATE
    if _DECODER_INIT_STATE is None:
        _DECODER_INIT_STATE = copy.deepcopy(net.decoder.state_dict())
    return _DECODER_INIT_STATE


def _fresh_decoder() -> nn.Module:
    """每次 run 重置 decoder，避免上一轮 AdaIN 权重残留。"""
    decoder = copy.deepcopy(net.decoder)
    decoder.load_state_dict(_get_decoder_init_state())
    return decoder


def train_transform():
    return transforms.Compose(
        [
            transforms.Resize(size=(512, 512)),
            transforms.RandomCrop(256),
            transforms.ToTensor(),
        ]
    )


class NumericImageFolder(ImageFolder):
    """数字类名目录：按数值 1..N 排序，标签为 0..N-1。"""

    def find_classes(self, directory: str):
        names = sorted(
            (d.name for d in Path(directory).iterdir() if d.is_dir() and d.name.isdigit()),
            key=lambda x: int(x),
        )
        if not names:
            raise FileNotFoundError(f"未在 {directory} 找到数字类别子目录")
        class_to_idx = {name: int(name) - 1 for name in names}
        return names, class_to_idx


def adjust_learning_rate(optimizer, lr, lr_decay, iteration_count):
    new_lr = lr / (1.0 + lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


@torch.no_grad()
def infer_feat_dim(vgg_module, device, sample_loader) -> int:
    imgs, _ = next(iter(sample_loader))
    _, g = vgg_module(imgs.to(device))
    return int(g.view(g.size(0), -1).size(1))


def train_linear_evaluator(vgg_module, train_loader, test_loader, device, num_classes, epochs=20, lr=1e-3):
    """冻结 VGG，在全局池化特征上训练线性分类头，返回测试集四项指标。"""
    vgg_module.eval()
    feat_dim = infer_feat_dim(vgg_module, device, train_loader)
    classifier = nn.Linear(feat_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        classifier.train()
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                _, g = vgg_module(imgs)
                feats = g.view(g.size(0), -1)
            logits = classifier(feats)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    classifier.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            _, g = vgg_module(imgs)
            feats = g.view(g.size(0), -1)
            pred = classifier(feats).argmax(dim=1)
            y_true.extend(labels.tolist())
            y_pred.extend(pred.cpu().tolist())

    labels_all = list(range(num_classes))
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)
    return RunMetrics(
        accuracy=float(np.mean(y_true_arr == y_pred_arr)),
        macro_f1=float(
            f1_score(y_true_arr, y_pred_arr, average="macro", labels=labels_all, zero_division=0)
        ),
        weighted_f1=float(
            f1_score(y_true_arr, y_pred_arr, average="weighted", labels=labels_all, zero_division=0)
        ),
        balanced_accuracy=float(balanced_accuracy_score(y_true_arr, y_pred_arr)),
    )


def print_test_metrics(metrics: RunMetrics, dataset_name: str, run_idx: Optional[int] = None) -> None:
    run_tag = f" run{run_idx}" if run_idx is not None else ""
    print("\n" + "=" * 60)
    print(f"{dataset_name} 测试集结果{run_tag}")
    print("=" * 60)
    print(f"Accuracy          : {metrics['accuracy']:.4f}")
    print(f"Macro-F1          : {metrics['macro_f1']:.4f}")
    print(f"Weighted-F1       : {metrics['weighted_f1']:.4f}")
    print(f"Balanced Accuracy : {metrics['balanced_accuracy']:.4f}")
    print("=" * 60)
    print(
        f"| Accuracy | Macro-F1 | Weighted-F1 | Balanced Accuracy |\n"
        f"| {metrics['accuracy']:.4f} | {metrics['macro_f1']:.4f} | "
        f"{metrics['weighted_f1']:.4f} | {metrics['balanced_accuracy']:.4f} |"
    )
    print("=" * 60 + "\n")


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
    max_iter: int,
    clf_epochs: int,
    runs: int,
    completed_runs: Optional[int] = None,
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
        f"(max_iter={max_iter}, clf_epochs={clf_epochs}, runs={runs}{progress}) — {timestamp}",
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
    all_runs: List[RunMetrics],
    max_iter: int,
    clf_epochs: int,
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
        max_iter,
        clf_epochs,
        total_runs,
        completed_runs=len(all_runs),
    )


def merge_batch_partials(
    partial_dir: str,
    merge_result_md: str,
    runs: int,
    data_base: str = DEFAULT_DATA_BASE,
    max_iter: int = 10000,
    clf_epochs: int = 20,
) -> None:
    """合并 partial_dir 下各数据集 md → 五库总表。"""
    if not data_base.endswith("/"):
        data_base += "/"

    partials: Dict[str, str] = {}
    for path in sorted(glob.glob(os.path.join(partial_dir, "*.md"))):
        name = os.path.splitext(os.path.basename(path))[0]
        with open(path, encoding="utf-8") as f:
            partials[name] = f.read()

    if partials:
        m = re.search(
            r"max_iter=(\d+),\s*clf_epochs=(\d+)",
            next(iter(partials.values())),
        )
        if m:
            max_iter, clf_epochs = int(m.group(1)), int(m.group(2))

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
        f"(max_iter={max_iter}, clf_epochs={clf_epochs}, runs={runs}) — "
        f"{datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        f"_data_base=`{data_base}`_",
        "",
        f"_命令: `./ST-SACLF-ncc_main/pytorch-AdaIN/run_st_saclf_train_bat.sh` → "
        f"`train.py` × {len(DATASET_ORDER)} 数据集_",
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


def train_single_run(args, device: torch.device, run_idx: int, dataset_name: str) -> RunMetrics:
    """单次 run：AdaIN 训练 + 线性分类评估。"""
    data_root = Path(args.data_root)
    train_dir = data_root / "train"
    test_dir = data_root / "test"
    if not train_dir.is_dir() or not test_dir.is_dir():
        raise FileNotFoundError(f"需要 {train_dir} 与 {test_dir} 目录")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    vgg = net.vgg

    decoder = _fresh_decoder()
    network = net.Net(decoder)
    network.train().to(device)

    adain_tf = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    content_dataset = NumericImageFolder(str(train_dir), transform=adain_tf)
    style_dataset = NumericImageFolder(str(train_dir), transform=adain_tf)

    if not args.skip_adain:
        log_dir = Path(args.log_dir) / f"{dataset_name}_run{run_idx}"
        log_dir.mkdir(exist_ok=True, parents=True)
        writer = SummaryWriter(log_dir=str(log_dir))

        content_iter = iter(
            data.DataLoader(
                content_dataset,
                batch_size=args.batch_size,
                sampler=InfiniteSamplerWrapper(content_dataset),
                num_workers=args.n_threads,
            )
        )
        style_iter = iter(
            data.DataLoader(
                style_dataset,
                batch_size=args.batch_size,
                sampler=InfiniteSamplerWrapper(style_dataset),
                num_workers=args.n_threads,
            )
        )
        optimizer = torch.optim.Adam(network.decoder.parameters(), lr=args.lr)

        for i in tqdm(range(args.max_iter), desc=f"AdaIN {dataset_name} run{run_idx}"):
            adjust_learning_rate(optimizer, args.lr, args.lr_decay, i)
            content_images, _ = next(content_iter)
            style_images, _ = next(style_iter)
            content_images = content_images.to(device)
            style_images = style_images.to(device)

            content_f, _ = vgg(content_images)
            style_f, _ = vgg(style_images)
            loss_c, loss_s = network(content_f, style_f)
            loss_c = args.content_weight * loss_c
            loss_s = args.style_weight * loss_s
            loss = loss_c + loss_s

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar("loss_content", loss_c.item(), i + 1)
            writer.add_scalar("loss_style", loss_s.item(), i + 1)

            if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
                torch.save(
                    network.decoder.state_dict(),
                    save_dir / f"decoder_iter_{i + 1}_{dataset_name}_run{run_idx}.pth",
                )
        writer.close()

    eval_tf = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    train_eval_ds = NumericImageFolder(str(train_dir), transform=eval_tf)
    test_eval_ds = NumericImageFolder(str(test_dir), transform=eval_tf)
    train_eval_loader = DataLoader(
        train_eval_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.n_threads
    )
    test_eval_loader = DataLoader(
        test_eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.n_threads
    )

    return train_linear_evaluator(
        vgg,
        train_eval_loader,
        test_eval_loader,
        device,
        args.num_classes,
        epochs=args.clf_epochs,
        lr=args.clf_lr,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=f"{MODEL_NAME} 训练 + 分类评估")
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT, help="数据集根目录")
    parser.add_argument("--num_classes", type=int, default=NUM_CLASSES)
    parser.add_argument("--vgg", type=str, default=str(_DEFAULT_VGG), help="vgg_normalised.pth 路径")
    parser.add_argument("--save_dir", default=str(_ST_SACLF_ROOT / "experiments" / "adain_decoders"))
    parser.add_argument("--log_dir", default=str(_SCRIPT_DIR / "logs"))
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_decay", type=float, default=5e-5)
    parser.add_argument("--max_iter", type=int, default=10000, help="AdaIN 训练迭代次数")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--style_weight", type=float, default=10.0)
    parser.add_argument("--content_weight", type=float, default=1.0)
    parser.add_argument("--n_threads", type=int, default=4)
    parser.add_argument("--save_model_interval", type=int, default=10000)
    parser.add_argument("--clf_epochs", type=int, default=20, help="线性分类头 epoch 数")
    parser.add_argument("--clf_lr", type=float, default=1e-3, help="线性分类头学习率")
    parser.add_argument("--skip_adain", action="store_true", help="跳过 AdaIN，仅做分类评估")
    parser.add_argument(
        "--benchmark_all",
        action="store_true",
        help="依次在五数据集上训练（data_base 下子目录同 simclr/barlowtwins）",
    )
    parser.add_argument("--data_base", type=str, default=DEFAULT_DATA_BASE)
    parser.add_argument("--result_md", type=str, default=DEFAULT_RESULT_MD)
    parser.add_argument(
        "--merge_result_md",
        type=str,
        default=None,
        help="批量模式：每轮 run 后合并 partial 写入此总表",
    )
    parser.add_argument(
        "--partial_dir",
        type=str,
        default=None,
        help="各数据集 partial md 所在目录（配合 --merge_result_md）",
    )
    parser.add_argument(
        "--dataset_label",
        type=str,
        default=None,
        help="批量脚本中的数据集显示名（如 ArtBench）",
    )
    parser.add_argument(
        "--run",
        "--runs",
        type=int,
        default=3,
        dest="num_runs",
        metavar="N",
        help="每个数据集重复次数（默认 3），记录 mean±std",
    )
    parser.add_argument(
        "--benchmark_runs",
        type=int,
        default=None,
        help="--benchmark_all 时重复次数；未指定时同 --runs",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("警告: 未检测到 CUDA，训练会很慢。")

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.result_md)), exist_ok=True)

    if args.num_runs < 1:
        raise SystemExit("错误: --runs 须 >= 1")

    benchmark_runs = args.benchmark_runs if args.benchmark_runs is not None else args.num_runs

    if args.benchmark_all:
        if benchmark_runs < 1:
            raise ValueError("--benchmark_runs 须 >= 1")
        results: List[DatasetResult] = []
        data_base = os.path.normpath(args.data_base)
        for rel, n_cls, label in BENCHMARK_DATASETS:
            args.data_root = os.path.join(data_base, rel.replace("/", os.sep))
            args.num_classes = n_cls
            all_runs: List[RunMetrics] = []
            for run_idx in range(1, benchmark_runs + 1):
                print(f"[{label}] run {run_idx}/{benchmark_runs} 开始…")
                try:
                    metrics = train_single_run(args, device, run_idx, label)
                    all_runs.append(metrics)
                    print_test_metrics(metrics, label, run_idx)
                    print(
                        f"[{label}] run {run_idx}/{benchmark_runs} "
                        f"acc={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}"
                    )
                except Exception as e:
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
            args.max_iter,
            args.clf_epochs,
            benchmark_runs,
        )
        print(f"结果已写入: {args.result_md}")
        return

    data_root = os.path.abspath(args.data_root.rstrip("/"))
    dataset_name = args.dataset_label or os.path.basename(os.path.normpath(data_root))
    all_runs: List[RunMetrics] = []

    for r in range(1, args.num_runs + 1):
        print(f"[{dataset_name} run{r}/{args.num_runs}] 开始训练…")
        try:
            metrics = train_single_run(args, device, r, dataset_name)
            all_runs.append(metrics)
            print_test_metrics(metrics, dataset_name, r)
            print(
                f"[{dataset_name} run{r}/{args.num_runs}] "
                f"acc={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}, "
                f"weighted_f1={metrics['weighted_f1']:.4f}, "
                f"balanced_acc={metrics['balanced_accuracy']:.4f}"
            )
        except Exception as e:
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
            args.max_iter,
            args.clf_epochs,
            args.num_runs,
        )
        print(f"结果已更新: {args.result_md} ({len(all_runs)}/{args.num_runs} runs)")

        if args.merge_result_md and args.partial_dir:
            merge_batch_partials(
                args.partial_dir,
                args.merge_result_md,
                args.num_runs,
                args.data_base,
                args.max_iter,
                args.clf_epochs,
            )
            print(f"总表已更新: {args.merge_result_md}")

    print(f"[{dataset_name}] Accuracy {_format_mean_std([m['accuracy'] for m in all_runs])}")


if __name__ == "__main__":
    main()
