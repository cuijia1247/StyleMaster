#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据 ieee_access_paperdata/ours_multiple.md 中各数据集最佳 run 的 checkpoint，
在测试集上绘制 SSC-ResNet50 混淆矩阵。

用法（项目根目录）::
  python ieee_access_codes/plot_ieee_ssc_confusion_matrix.py
  python ieee_access_codes/plot_ieee_ssc_confusion_matrix.py --datasets Pandora Arch
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ssc.utils import MultiViewDataInjector, get_ssc_transforms  # noqa: E402
from SscDataSet_new import SscDataset  # noqa: E402
from ssc_train_resnet_copy import parameter_load  # noqa: E402
from utils.pretrainFeatureExtraction import load_dataFeatures  # noqa: E402

DEFAULT_DATA_BASE = "/mnt/codes/data/style"
DEFAULT_MODEL_DIR = os.path.join(_ROOT, "model")
DEFAULT_PRE_FEATURE = os.path.join(_ROOT, "pretrainFeatures")
DEFAULT_OUT_DIR = os.path.join(_ROOT, "ieee_access_paperdata")

# ours_multiple.md 中 Accuracy 列（run1/run2/run3），用于选取最佳 run
BENCHMARK_RUN_ACCURACY: Dict[str, Tuple[float, float, float]] = {
    "Painting91": (0.7479, 0.7605, 0.7185),
    "Pandora": (0.6179, 0.6107, 0.6133),
    "FashionStyle14": (0.7003, 0.7015, 0.6996),
    "Arch": (0.6794, 0.6809, 0.6774),
}

DATASET_CFG: Dict[str, Tuple[str, int]] = {
    "Painting91": ("Painting91", 13),
    "Pandora": ("Pandora", 12),
    "FashionStyle14": ("FashionStyle14", 14),
    "Arch": ("Arch", 25),
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@dataclass
class CheckpointPair:
    base_path: str
    classifier_path: str
    run_idx: int
    acc_suffix: int


def _best_run_index(accs: Sequence[float]) -> int:
    return int(np.argmax(accs)) + 1


def _parse_acc_suffix(path: str) -> int:
    m = re.search(r"-accuracy-(\d+)-SSC-classifier-best\.pth$", os.path.basename(path))
    if not m:
        raise ValueError(f"无法解析 checkpoint 精度后缀: {path}")
    return int(m.group(1))


def find_best_checkpoint(
    model_dir: str, dataset_label: str, ieee_run_no: int
) -> CheckpointPair:
    """ieee run 编号 1-based；模型文件名 run 后缀为 0-based。"""
    run_suffix = f"-run{ieee_run_no - 1}-"
    pattern = os.path.join(
        model_dir,
        f"ssc-{dataset_label}-SSC-resnet50-*{run_suffix}*-SSC-classifier-best.pth",
    )
    candidates = sorted(glob.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"未找到 checkpoint: {pattern}")

    best_path = max(candidates, key=_parse_acc_suffix)
    acc_suffix = _parse_acc_suffix(best_path)
    base_path = best_path.replace("-SSC-classifier-best.pth", "-SSC-base-best.pth")
    if not os.path.isfile(base_path):
        raise FileNotFoundError(f"缺少配套 base 模型: {base_path}")
    return CheckpointPair(
        base_path=base_path,
        classifier_path=best_path,
        run_idx=ieee_run_no,
        acc_suffix=acc_suffix,
    )


def infer_class_names(data_root: str, num_classes: int) -> List[str]:
    """从 test/ 子目录首样本文件名推断类别名。"""
    names: List[str] = []
    test_root = os.path.join(data_root, "test")
    for cls_id in range(1, num_classes + 1):
        folder = os.path.join(test_root, str(cls_id))
        if not os.path.isdir(folder):
            names.append(f"C{cls_id}")
            continue
        sample = sorted(os.listdir(folder))[0]
        stem = os.path.splitext(sample)[0]
        stem = re.sub(r"_\d+$", "", stem)
        names.append(stem.replace("_", " "))
    return names


@torch.no_grad()
def predict_test_set(
    model: torch.nn.Module,
    classifier: torch.nn.Module,
    data_source: str,
    feature_dict: dict,
    image_size: int,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    transform_t, transform_t1, transform_eval = get_ssc_transforms(
        image_size, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    )
    # 测试集混淆矩阵使用确定性 CenterCrop（两路视图相同），便于论文展示
    testset = SscDataset(
        data_source,
        "test",
        transform=MultiViewDataInjector([transform_eval, transform_eval]),
    )
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    model.eval()
    classifier.eval()
    y_true: List[int] = []
    y_pred: List[int] = []

    for view1, view2, label, names, _ in testloader:
        view1 = view1.to(device)
        view2 = view2.to(device)
        backbone_view = torch.stack([feature_dict[n] for n in names], dim=0).to(device)
        ssc_v1 = model(view1)
        ssc_v2 = model(view2)
        logits = classifier(ssc_v1, ssc_v2, backbone_view)
        pred = logits.argmax(dim=1)
        y_true.extend((label - 1).long().cpu().tolist())
        y_pred.extend(pred.cpu().tolist())

    return np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)


def plot_and_save_cm(
    cm: np.ndarray,
    class_names: List[str],
    title: str,
    out_path: str,
    accuracy: float,
) -> None:
    # 归一化到行（真实类别）便于观察召回
    cm_norm = cm.astype(np.float64)
    row_sum = cm_norm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_norm, row_sum, where=row_sum > 0)

    n_cls = len(class_names)
    fig_size = max(8.0, n_cls * 0.55)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.92))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_step = 1 if n_cls <= 20 else 2
    ticks = list(range(0, n_cls, tick_step))
    short_names = [
        (name if len(name) <= 14 else name[:12] + "…") for name in class_names
    ]
    ax.set(
        xticks=ticks,
        yticks=ticks,
        xticklabels=[short_names[i] for i in ticks],
        yticklabels=[short_names[i] for i in ticks],
        ylabel="True label",
        xlabel="Predicted label",
        title=f"{title}\nTest Acc={accuracy:.4f} (normalized by true class)",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = 0.5
    for i in range(n_cls):
        for j in range(n_cls):
            val = cm[i, j]
            if val == 0:
                continue
            color = "white" if cm_norm[i, j] > thresh else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存: {out_path}")


def process_dataset(
    label: str,
    rel_dir: str,
    num_classes: int,
    data_base: str,
    model_dir: str,
    pre_feature_path: str,
    out_dir: str,
) -> None:
    accs = BENCHMARK_RUN_ACCURACY[label]
    best_run = _best_run_index(accs)
    ckpt = find_best_checkpoint(model_dir, label, best_run)

    params = parameter_load()
    _, batch_size, _, _, image_size, *_ = params

    data_source = os.path.join(data_base.rstrip("/"), rel_dir).rstrip("/") + "/"
    feature_name = f"{rel_dir.replace('/', '_')}_resnet50"
    test_feature_path = os.path.join(pre_feature_path, f"{feature_name}_test_features.pkl")
    test_feature_dict = load_dataFeatures(test_feature_path)

    print(
        f"[{label}] 选用 run{best_run} (md acc={accs[best_run - 1]:.4f}), "
        f"checkpoint acc_suffix={ckpt.acc_suffix / 10000:.4f}"
    )
    print(f"  classifier: {ckpt.classifier_path}")
    print(f"  base:       {ckpt.base_path}")

    model = torch.load(ckpt.base_path, map_location=device)
    classifier = torch.load(ckpt.classifier_path, map_location=device)
    model = model.to(device)
    classifier = classifier.to(device)

    y_true, y_pred = predict_test_set(
        model, classifier, data_source, test_feature_dict, image_size, batch_size
    )
    accuracy = float(np.mean(y_true == y_pred))
    print(f"  测试集 Accuracy={accuracy:.4f}, N={len(y_true)}")

    class_names = infer_class_names(data_source.rstrip("/"), num_classes)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    out_png = os.path.join(out_dir, f"ours_ssc_confusion_matrix_{label}_test.png")
    plot_and_save_cm(
        cm,
        class_names,
        title=f"Ours (SSC-ResNet50) — {label} (run{best_run})",
        out_path=out_png,
        accuracy=accuracy,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="绘制 SSC-ResNet50 测试集混淆矩阵")
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["Painting91", "FashionStyle14"],
        choices=list(DATASET_CFG.keys()),
    )
    p.add_argument("--data_base", type=str, default=DEFAULT_DATA_BASE)
    p.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR)
    p.add_argument("--pre_feature_path", type=str, default=DEFAULT_PRE_FEATURE)
    p.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    for label in args.datasets:
        rel_dir, num_classes = DATASET_CFG[label]
        process_dataset(
            label,
            rel_dir,
            num_classes,
            args.data_base,
            args.model_dir,
            args.pre_feature_path,
            args.out_dir,
        )


if __name__ == "__main__":
    main()
